# Deep Learning Based Building Extraction
This repository contains notebooks and scripts used for ESA-CEOS training on Deep learning based building extraction from Very High Resolution Imagery
There are three main important topics covered in this practical

- SAM3 interactive building extraction (instance segmentation
- Dataset preparation for object detection and instance segmentation
- Training and inference of building object detection and instance segmentation with YOLO
- Post processing, converting model predictions to geospatial building object layers

Some pre- and post-processing functions are included as a Python standalone script and embedded in the notebook. 


# The Training Data
The data used in this training is accessed from [Open Aerial Maps](https://openaerialmap.org/), which is taken from the [Kakuma Refugee Camp](https://map.openaerialmap.org/#/34.83318328857422,3.7289397744724693,12?_k=le5kmn). More information on the location can be accessed from resources([UNHCR Kena](https://www.unhcr.org/ke/about-us/where-we-work/kakuma-refugee-camp), [EU comission](https://civil-protection-humanitarian-aid.ec.europa.eu/news-stories/stories/life-kakuma-hope-and-resilience-refugee-camp_en), [IOM kena](https://kenya.iom.int/kakuma)). For the purpose of executing this tutorial, selected drone images are downloaded and hosted in Kaggle, which provides storage and fast downloads. Access scripts are included in the notebooks. If you want to run this demo or need the data for later use in any workflow, please download it to your local computer from [THIS LINK](https://www.kaggle.com/datasets/getachewworkineh/kakuma-ceos-training)

# Computational environment
The notebooks are designed to run on [Google Colab](https://colab.research.google.com/), which also provides free basic computing resources, but for very high projects that demand high-performance resources, one could consider either a subscription-based or other local resources. 
Please note that while running the training and inference, try to chnage the runtime to GPU resources. More information on usage and resource optimisation in  [Google Colab can be accessed here](https://colab.research.google.com/)

# Models
- SAM3: State-of-the-art instance model for segmentation, detection and tracking. More information on the model can be accessed from [Meta's AI research site](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/), and the implementation details, including training and benchmarking, can be accessed from [here](https://arxiv.org/abs/2511.16719). This tutorial makes use of the SAM3 embedded in a [segment-geospatial](https://github.com/opengeos/segment-geospatial), which gives more flexible interaction.
- YOLO26: Its lightweight and fast model for classification, detection, and instance segmentation. Details about architectural improvement from predecessor versions can be found here [resource1](https://arxiv.org/pdf/2509.25164) and [resource2](https://arxiv.org/pdf/2602.14582), and usage, pretrained wights and further details can be accessed [here](https://docs.ultralytics.com/models/yolo26/#overview).

Please note that these models treat Earth observation images as any image, and there should always be a post-processing phase where model outputs will be converted to geospatial data. 

# Creation of Environment
The installation could be done using pixi as follows 

```bash
curl -fsSL https://pixi.sh/install.sh | sh```

Then close and re-open your terminal (or reload your shell) so pixi is on your PATH. Then confirm its proper installation by:

```pixi --version```

Then initialise the pixi environment using 

```
pixi init extract

cd extract
```

then edit the ```pixi.toml``` file to reflect packages we install as:

```
[workspace]
channels = ["https://prefix.dev/conda-forge"]
name = "geo"
platforms = ["linux-64", "win-64"]

[system-requirements]
cuda = "12.0"

[dependencies]
python = "3.12.*"
pytorch-gpu = ">=2.7.1,<3"
segment-geospatial = ">=1.2.0"
sam3 = ">=0.1.0.20251211"
jupyterlab = "*"
ipykernel = "*"
libopenblas = ">=0.3.30"
kagglehub = "*"
fiona = "*"
ultralytics = "*"
```

Then install the environment as:

```
pixi install
```
Then, verify proper installation by running the following in the terminal

```
pixi run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'))"
```

To properly use the installed pixi environment, please run the following in the terminal

```
pixi powershell
```

If you want to install an additional package, please use

```
pixi add your-package-name
```

For different CUDA and CPU versions, please refer original SamGeo installation [Here](https://samgeo.gishub.org/installation/)

# Usage
- Please download this repository, unzip it or use the command line interface 

```
git clone  https://github.com/getch-geohum/Deep-Learning-Building-Extraction.git
cd Deep-Learning-Building-Extraction
```

Then type the following and hit enter, which will redirect to Jupyterlab environment, navigate to **notebooks** folder and run the cells

```
jupyter lab
```
  
- SAM3 weights are only accessible from [Hugging Face](https://huggingface.co/) with registration. If you do not have an account on Hugging Face, you have to create an account and request usage for SAM3. Once it is approved, create a token in your Hugging Face account and paste it when it's required. This is mainly for SAM3 interactive segmentation.


