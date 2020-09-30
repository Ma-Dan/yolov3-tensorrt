# Dockerfile
FROM nvcr.io/nvidia/l4t-base:r32.4.3

# Install requirements
RUN apt-get update
RUN apt-get install -y git python3-pip cmake protobuf-compiler libprotoc-dev libopenblas-dev gfortran libjpeg8-dev libxslt1-dev libfreetype6-dev
RUN pip3 install Cython numpy
RUN pip3 install onnx==1.4.1 pycuda==2019.1 wget>=3.2 \
    Pillow>=5.2.0 bistiming eyewitness==1.1.1 scipy==1.2.1 celery==4.3.0 \
    gevent line-bot-sdk==1.8.0 fbchat==1.8.3 psutil
RUN pip3 install scikit-build && pip3 install opencv-python

# Get model and weights
RUN cd / && git clone https://github.com/Ma-Dan/yolov3-tensorrt && \
    cd /yolov3-tensorrt && wget https://kubeedge.pek3b.qingstor.com/yolov3.engine

EXPOSE 8001
