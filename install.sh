# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install -y --no-install-recommends nvidia-driver-418
# Reboot. Check that GPUs are visible using the command: nvidia-smi

# Install development and runtime libraries (~4GB)
sudo apt-get install -y --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.2.24-1+cuda10.0  \
    libcudnn7-dev=7.6.2.24-1+cuda10.0


# clean up stuff
sudo rm cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo rm nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

# Install python deps
sudo apt-get update
sudo apt install -y python3-pip
sudo apt-get install -y libsm6 libxext6 libxrender-dev
pip3 install tensorflow-gpu==1.14  # GPU
pip3 install opencv-python
pip3 install scipy
pip3 install scikit-learn
pip3 uninstall --yes numpy
pip3 install numpy==1.16.2

## Set env vars
export PUBLIC_DNS_NAME=`curl -s http://169.254.169.254/latest/meta-data/public-hostname`
export PUBLIC_IP=`curl http://169.254.169.254/latest/meta-data/public-ipv4`

echo "export PUBLIC_IP=\`curl http://169.254.169.254/latest/meta-data/public-ipv4\`" >> ~/.bashrc
echo "export PUBLIC_DNS_NAME=\`curl -s http://169.254.169.254/latest/meta-data/public-hostname\`" >> ~/.bashrc

## Install docker and docker-compose
sudo addgroup --system docker
sudo adduser $USER docker
newgrp docker

sudo snap install docker
sudo setfacl -m user:$USER:rw /var/run/docker.sock

sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose


## Build docker image
# docker build -t daemon .