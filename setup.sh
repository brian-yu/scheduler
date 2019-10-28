
## Install docker + docker-compose

sudo addgroup --system docker
sudo adduser $USER docker
newgrp docker

sudo snap install docker
sudo setfacl -m user:$USER:rw /var/run/docker.sock

sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose


## Install NVIDIA container support
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo snap stop docker
sudo snap start docker


## Build docker image
docker build -t daemon .


## Set env vars
export PUBLIC_DNS_NAME=`curl -s http://169.254.169.254/latest/meta-data/public-hostname`
export PUBLIC_IP=`curl http://169.254.169.254/latest/meta-data/public-ipv4`

echo "export PUBLIC_IP=\`curl http://169.254.169.254/latest/meta-data/public-ipv4\`" >> ~/.bashrc
echo "export PUBLIC_DNS_NAME=\`curl -s http://169.254.169.254/latest/meta-data/public-hostname\`" >> ~/.bashrc