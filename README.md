# Reconstruct-3D-Human-Pose
**Group Members: Xianghan Kong, Yangfei Gao**

## 1. About The Project
This project is our advanced computer vision course group prject. We explore the method of SPIN [https://arxiv.org/pdf/1909.12828.pdf](https://arxiv.org/pdf/1909.12828.pdf) and make some improvements. An example of the results is shown as below.

<img height='500' width='150' src='https://github.com/kxhaaa/Reconstruct-3D-Human-Pose/blob/main/examples/figure3.png'>

## 2. Getting Started

### Pakages Requirement
numpy, opencv-python, pyopengl, pyrender, scikit-image, scipy, chumpy, smplx, spacepy, pytorch, torchgeometry, torchvision, tqdm, trimesh, yolox

### Data Requirement
Model: [http://visiondata.cis.upenn.edu/spin/data.tar.gz](http://visiondata.cis.upenn.edu/spin/data.tar.gz)

initial_fits: [http://visiondata.cis.upenn.edu/spin/static_fits.tar.gz](http://visiondata.cis.upenn.edu/spin/static_fits.tar.gz)

pretrained_checkpoints: [http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt](http://visiondata.cis.upenn.edu/spin/model_checkpoint.pt)

GMM priors: [https://github.com/vchoutas/smplify-x/raw/master/smplifyx/prior.py](https://github.com/vchoutas/smplify-x/raw/master/smplifyx/prior.py)

SMPL and SMPL-X model: [https://smpl-x.is.tue.mpg.de/](https://smpl-x.is.tue.mpg.de/)

3DPW datasets: [https://virtualhumans.mpi-inf.mpg.de/3DPW/](https://virtualhumans.mpi-inf.mpg.de/3DPW/)

LSP: [http://sam.johnson.io/research/lsp.html](http://sam.johnson.io/research/lsp.html)

LSPET: [http://sam.johnson.io/research/lspet.html](http://sam.johnson.io/research/lspet.html)

### Usage
Run `python 8501_train_pipline.py` for training, `python 8501_evaluation.py` for evaluation, `python 8501_prediction.py` for prediction with single person, `python 8501_multi_prediction.py` for prediction with multiple people.
Run `python bboxes_by_YOLOX.py` for getting object's boundingbox information
## 3. Contribution
Xianghan Kong - 3D human estimation parts.  
Yangfei Gao - human detection parts.  

## Acknowledgement
We also thank Prof. Hongdong Li for ENGN 8501 and all the course tutors.








