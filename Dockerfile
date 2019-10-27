FROM tensorflow/tensorflow:1.14.0-gpu-py3  

WORKDIR /root
COPY . .

RUN ls

RUN pip3 uninstall --yes numpy
RUN pip3 install numpy==1.16.2
RUN pip3 install opencv-python
RUN pip3 install scipy
RUN pip3 install scikit-learn
RUN apt install -y netcat
RUN apt install -y libsm6 libxext6 libxrender-dev

CMD [ "python3 worker_daemon.py --port=8888" ]    