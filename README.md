# Deep Learning Based Building Extraction
This repository contains notebooks and scripts used for ESA-CEOS training on Deep learning based building extraction from Very High Resolution Imagery
There are three main important topics covered in this practical

- SAM3 interactive building extraction (instance segmentation
- Dataset preparation for object detection and instance segmentation
- Training and inference of building object detection and instance segmentation using YOLO
- Post processing, converting model predictions to geospatial building object layers

Some pre-and post processing functions are included as a python standalon script and embeded in the notebook. 


# The Training Data
The data used in this training is accessed from [Open Aerial Maps](https://openaerialmap.org/), which is taken from the [Kakuma IDP camp](https://map.openaerialmap.org/#/34.83318328857422,3.7289397744724693,12?_k=le5kmn). For the purpose of executing this tutorial, selected drone images are downloaded and hosted in Kaggle, which provides storage and fast downloads. Access scripts are included in the notebooks. If you want to run this demo or need the data for later use in any workflow, please download it to your local computer from [THIS LINK](https://www.kaggle.com/datasets/getachewworkineh/kakuma-ceos-training)

# Computational environment
The notebooks are designed to run on [Google Colab](https://colab.research.google.com/), which also provides free basic computing resources, but for very high projects that demand high-performance resources, one could consider either a subscription-based or other local resources. 
Please note that while running the training and inference, try to chnage the runtime to GPU resources. 

