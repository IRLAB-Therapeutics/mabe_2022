FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install pandas scipy scikit-learn
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install plotly
RUN pip install openpyxl
RUN pip install umap-learn
RUN pip install matplotlib
RUN pip install simclr aicrowd-cli
RUN pip install opencv-python
RUN pip install jupyter
RUN apt-get install unzip -y
RUN pip install transformers
RUN apt-get -y install git
RUN pip install pytest
RUN pip install pytorch-lightning
RUN pip install scikit-image