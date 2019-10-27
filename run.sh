# docker build -t daemon .
docker run --network host -v /home/ubuntu/scheduler:/root daemon python3 worker_daemon.py --port=8888