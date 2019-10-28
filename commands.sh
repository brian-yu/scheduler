

kill -9 `ps -ef | grep ps.py | awk '{print $2}'`
1.14.0-gpu-py3

echo "START_PS test localhost:2222 localhost:2223" | nc localhost 8889


sudo setfacl -m user:$USER:rw /var/run/docker.sock



sudo curl -L "https://github.com/docker/compose/releases/download/1.24.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

echo "START_PS test ps0:2222 worker0:2223" | nc ps0 8888
echo "TRAIN test ps0:2222 worker0:2223" | nc worker0 8888


echo "START_PS test1 ps0:2223 worker1:2222" | nc ps0 8888
echo "TRAIN test1 ps0:2223 worker1:2222" | nc worker1 8888


docker exec worker0 cat checkpoints/test/checkpoint
docker exec ps0 ls checkpoints/test/