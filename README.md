# Deep Learning Based Building Extraction
This repository contains notebooks and scripts used for ESA-CEOS training on Deep learning based building extraction from Very High Resolution Imagery
There are three main important topics covered in this practical

- SAM3 interactive building extraction (instance segmentation
- Dataset preparation for object detection and instance segmentation
- Training and inference of building object detection and instance segmentation with YOLO
- Post processing, converting model predictions to geospatial building object layers

Some pre- and post-processing functions are included as a Python standalone script and embedded in the notebook. 


# The Training Data
The data used in this training is accessed from [Open Aerial Maps](https://openaerialmap.org/), which is taken from the [Kakuma IDP camp](https://map.openaerialmap.org/#/34.83318328857422,3.7289397744724693,12?_k=le5kmn). For the purpose of executing this tutorial, selected drone images are downloaded and hosted in Kaggle, which provides storage and fast downloads. Access scripts are included in the notebooks. If you want to run this demo or need the data for later use in any workflow, please download it to your local computer from [THIS LINK](https://www.kaggle.com/datasets/getachewworkineh/kakuma-ceos-training)

# Computational environment
The notebooks are designed to run on [Google Colab](https://colab.research.google.com/), which also provides free basic computing resources, but for very high projects that demand high-performance resources, one could consider either a subscription-based or other local resources. 
Please note that while running the training and inference, try to chnage the runtime to GPU resources. More information on usage and resource optimisation in  [Google Colab can be accessed here](https://colab.research.google.com/)

# Models
- SAM3: State-of-the-art instance model for segmentation, detection and tracking. More information on the model can be accessed from [Meta's AI research site](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), and the implementation details, including training and benchmarking, can be accessed from [here](https://arxiv.org/abs/2511.16719). This tutorial makes use of the SAM3 embedded in a [segment-geospatial](https://github.com/opengeos/segment-geospatial), which gives more flexible interaction.
- YOLO26: Its lightweight and fast model for classification, detection, and instance segmentation. Details about architectural improvement from predecessor versions can be found here [resource1](https://arxiv.org/pdf/2509.25164) and [resource2](https://arxiv.org/pdf/2602.14582), and usage, pretrained wights and further details can be accessed [here](https://docs.ultralytics.com/models/yolo26/#overview).

Please note that these models treat Earth observation images as any image, and there should always be a post-processing phase where model outputs will be converted to geospatial data. 

# Usage 

Please download this repository, unzip and load it to [Google Drive](https://drive.google.com/), which is always easy to bridge with Google Colab. Then, to execute, double-click the Notebooks, and it will redirect to Google Colab. If you want to run in your local environment, please consider installing neccessary packages.
