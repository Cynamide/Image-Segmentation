# Image-Segmentation with U-net implementation using Tensorflow
<p align="center" height=50px><img src="https://i.imgur.com/uPFlRXT.gif" height="500"> </p> 
<p align="center">Demonstration of a use case of this technology to automatically blur computer screens while taking a video saving any sensitive information</p>

# Overview
This is my [TensorFlow](https://www.tensorflow.org/) implementations of U-net as a model for this Image Segmentation Project as inspired from
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) and [Image Segmentation in Tensorflow](https://www.tensorflow.org/tutorials/images/segmentation?hl=en)

## Model
<img src="https://i.imgur.com/UoCegKZ.png">

- This deep neural network is implemented with Tensorflow functional API, which makes it extremely easy to experiment with different interesting architectures.
- Output from the network is a 224*224 image which represents mask that should be learned.
- Transfer Learning is used by having MobileNetV2 as the Encoding stack of model. 

## Training

- The model is trained for 10 epochs.
- After 10 epochs, calculated accuracy is approximately about 92%.
- Loss function for the training is a binary crossentropy.

<img src="https://i.imgur.com/puAtRH7.png">

### Loss Plot

![image](https://user-images.githubusercontent.com/56030842/124399003-635ddb80-dd36-11eb-9fb9-b704b9f06558.png)


## Prerequisites

- Python 3.7
- [Tensorflow 2.5.x](https://github.com/tensorflow/tensorflow/)
- [NumPy](http://www.numpy.org/)
- [PIL](https://pillow.readthedocs.io/en/stable/)
- [Matplotlib](https://matplotlib.org/)
<br>
Clone this repo.
 
```bash
   git clone https://github.com/Cynamide/Image-Segmentation.git
``` 

Python in your machine should be 3.7.x.<br>
Install Jupyter Notebook.<br>
```bash
   pip install jupyter-notebook
``` 

# Running the Notebook

## Preparing the Dataset

This notebook is trained on the [COCO Dataset](https://cocodataset.org/#home). To run the notebook, first you will need to downlod their dataset from:

- [2017 Train Images](http://images.cocodataset.org/zips/train2017.zip)
- [2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip)
- [Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

Now make a new Folder named as "COCOdataset2017" with the file structure matching as shown below:

```bash
.
├──...
├── make_dataset.py                   
├── train.py
├── README.md
├──COCOdataset2017
   ├──annotations(.json files go here)
   ├──images
      ├──train
      ├──val
      ├──train_mask(empty directory)
      ├──train_img(empty directory)
```
## Trianing
After creating the file structure as above, simply run the make_dataset.ipynb notebook to:
- filter images of required annotations
- generate masks
- resize masks and images  
- save the images and masks in train_img and train_mask (refer the file structure)

Next and the final step is to run the next notebook train.py to train the model and look at the predictions it has made. You can also create your own masked video by runnig the code cell at the end of the notebook.

