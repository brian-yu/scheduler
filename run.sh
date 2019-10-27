docker build -t daemon .
docker run --network host -v .:. daemon python3 worker_daemon.py --port=8888