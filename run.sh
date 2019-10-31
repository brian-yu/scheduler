# Shutdown currently running FTP server.
docker-compose down
# Start FTP server in background.
docker-compose up -d

python3 worker_daemon.py --port=8888