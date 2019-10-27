

sudo snap install docker
sudo setfacl -m user:$USER:rw /var/run/docker.sock

sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose


docker build -t daemon .

export PUBLIC_DNS_NAME=`curl -s http://169.254.169.254/latest/meta-data/public-hostname`
