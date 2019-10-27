docker build -t daemon .
docker run --network host daemon python3 worker_daemon.py --port=8888