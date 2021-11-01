
'''
This interface is using YOLOX-Darknet to inference the bounding box information for SPIN, it will generate the json file
to store the bounding box information. Or you can also use this inferface to get the list type of the boundingbox information.
You need to download this file https://pan.baidu.com/s/1WOGfN284hKZnAjYZcnLfHQ by using password:af1f and put the file into the root folder.
You also need to setup the environment of yolox by https://github.com/Megvii-BaseDetection/YOLOX.

'''
import os
import time
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess
import json

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

#load the file under the defined path
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

#this class is for inference
class Predictor(object):
    def __init__(
        self,
        model,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = 80
        self.confthre = 0.5
        self.nmsthre = 0.65
        self.test_size = 640
        self.device = device

        self.preproc = ValTransform(legacy=False)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_standard_imgsize[0], test_standard_imgsize[1]).cuda()
            self.model(x)
            self.model = model_trt
    #This is main part for inference. We need to resize the image to put it into the YOLOX model
    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size / img.shape[0], self.test_size / img.shape[1])
    
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, [self.test_size, self.test_size])
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        return outputs, img_info
    #after we get the detection results, we should do some filter to these results. We pick the human detecion result
    #and hight confidence results to return and store.
    def visual(self, output, img_info, cls_conf=0.5):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        bboxall = []
        for i in range(len(bboxes)):
            score = scores[i]
            if score < cls_conf:
                continue
            if cls[i] != 0:
                continue
            box = bboxes[i]
            bx = box.tolist()
            #we should set [x_min, y_min, x_max, y_max] into form [x_min, y_min, weight, height]
            bx[2] = bx[2] - bx[0]
            bx[3] = bx[3] - bx[1]
            bboxall.append(bx)
        
        return bboxall 

# This function use the above method to get a whole pipeline of the inference
def image_inference(predictor, path, current_time):
    #get whole images list
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    #Iterative processing of all pictures
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        bboxes = predictor.visual(outputs[0], img_info, predictor.confthre)
        
    return bboxes

# This is the method which can be used by import in orther file, it will store the json file and return the boundingbox result
def get_bboxes(path = 'test_img2.png', pretrained_model = 'yolox_darknet.pth', exp = get_exp('yolov3.py', None)):
    model_name = 'yolov3'
    model = exp.get_model()
    
    model.cuda()

    model.eval()

   
    inference_model = torch.load(pretrained_model, map_location="cpu")
    # load the model state dict
    model.load_state_dict(inference_model["model"])
    #logger.info("loaded checkpoint done.")

    
    trt_file = None
    decoder = None

    predictor = Predictor(model, COCO_CLASSES, trt_file, decoder, 'gpu')
    current_time = time.localtime()
    bboxes = image_inference(predictor, path, current_time)
    #we use json package to store the bounding box information
    dic_bbox = {'bbox': bboxes}
    with open("bbox.json", "w") as f:
        json.dump(dic_bbox, f)
    return bboxes


if __name__ == "__main__":
    get_bboxes()