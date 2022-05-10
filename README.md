# MuRET's User Training Tool
[![GitHub Repo stars](https://img.shields.io/github/stars/JuanCarlosMartinezSevilla/MuRET-UserTool)](https://github.com/JuanCarlosMartinezSevilla/MuRET-UserTool/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/JuanCarlosMartinezSevilla/MuRET-UserTool)](https://github.com/JuanCarlosMartinezSevilla/MuRET-UserTool/watchers) 
[![GitHub last commit](https://img.shields.io/github/last-commit/JuanCarlosMartinezSevilla/MuRET-UserTool)](https://github.com/JuanCarlosMartinezSevilla/MuRET-UserTool/commits/main)


Most of the musical heritage is only available as physical documents, given that the engraving process was carried out by handwriting or typesetting until the end of the 20th century. Their mere availability as scanned images does not enable tasks such as indexing or editing unless they are transcribed into a structured digital format. The transcription process from historical handwritten music manuscripts to a structured digital encoding has been traditionally performed following a fully manual workflow. At most, it has received some technological support in particular stages, like optical music recognition (OMR) of the source images, or transcription to modern notation with music edition applications.

A new online tool called MUsic Recognition, Encoding, and Transcription (MuRET) has been recently developed, which covers all transcription phases, from the manuscript image to the encoded digital content. MuRET is designed as a machine-learning based research tool, allowing different processing approaches to be used, and producing both the expected transcribed contents in standard encodings and data for the study of the transcription process.

The objective of this repository is to provide the users a simple way to train deep learning models allowing an efficient transcription process.

## Why?
This tool is capable of training 3 classifiers:
- End to end staff classifier: Receives a staff image and returns all the ordered symbols that appear.
- Document analysis: Receives a full page image and returns a binarized image with its staves position.
- Symbol classifier: Receives a symbol cropped image and returns the symbol's type and position in the staff (line or space *x*).


When the training is done, it generates a [MuRET](https://muret.dlsi.ua.es/muret/#/about) package with all the required files. You can upload it to [MuRET](https://muret.dlsi.ua.es/muret/#/about) and see them in action.


### Installation

In order to use this tool, install [python](https://www.python.org/downloads/) and [CUDA](https://developer.nvidia.com/cuda-downloads). 

Second step is to clone this repository.
```shell
git clone https://github.com/JuanCarlosMartinezSevilla/MuRET-UserTool.git
```

Next, inside the project folder (MuRET-UserTool/), create a Virtual Environment. 
More info at https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment .

```shell
python3 -m venv .venv
```
Once created, activate it and install package manager pip.

```shell
source .venv/bin/activate
pip install --upgrade pip
```
Next step is to introduce all the dependencies.
```shell
python3 -m pip install -r requirements.txt
```
### Usage
You have options when executing the program.
```shell
Usage: python3 main.py [OPTIONS]

Options:
  -h,     --help              Shows this help message and exits
  -p,     --path              Path to dataset .tgz file.
  -pkg,   --pkg_name          Name of generated package.
  -da,    --doc_analysis      Train a document analysis model.
  -e2e,   --end_to_end        Train an agnostic end to end model.
  -sc,    --symb_classifier   Train a symbol classifier model.
  -rl,    --reload            Reloads dataset from MuRET (first execution you always need to use it).
  -ni,    --new_images        Number of new synthetic images (if required).
  -h5,    --h5                Save models .h5 format.
  
  You can include multiple paths:
    -p /path1/data.tgz /path2/data.tgz /path3/data.tgz
  
  Example:
    python3 main.py -p MuRETDatasets/mensural_manuscript.tgz -pkg tutorial -da -e2e -sc -ni 20 -rl
```
When finalized you will find your *MuRETPackage.tgz* (in this case the name will be *tutorial.tgz*) file in the same folder as the project. 
The structure inside this .tgz file is:

<p align="center">
  <img src="https://user-images.githubusercontent.com/97530443/167580141-2de57f01-72b6-42f9-84f6-e0ff309f529a.png" width="322" height="468">
</p>



Upload it to MuRET and see the magic happen.

### In case you don't have a good setup
This section allows users without a powerful machine to train the different deep learning models the same way this repository does. To achieve this purpose we have prepared a python notebook in Google Colaboratory that simplifies the process. However, this method is slower than using a desktop GPU, but it is completely free. Here you have it:

- [MuRET_Tutorial.ipynb](https://colab.research.google.com/drive/1Fu5zTnb57h20ymINXAc-UsdFc__YD-YR?usp=sharing)
<br>
  
Thank you for using **MuRET's User Training Tool**
