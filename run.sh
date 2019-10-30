# docker run --network host --name daemon -it -v /home/ubuntu/scheduler:/root daemon python3 worker_daemon.py --port=8888


docker-compose down
docker-compose up -d

python3 worker_daemon.py --port=8888