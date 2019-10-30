from enum import Enum
import time

class Command(Enum):
    TRAIN = "TRAIN"
    VALIDATE = "VALIDATE"
    TEST = "TEST"
    START_PS = "START_PS"
    STOP_PS = "STOP_PS"
    POLL = "POLL"
    RESET = "RESET"

class Status(Enum):
    FREE = "FREE"
    BUSY = "BUSY"

class Event(Enum):
    DOWNLOAD = "DOWNLOAD"
    BUILD = "BUILD"
    TRAIN = "TRAIN"
    SAVE = "SAVE"

class Logger:
    def __init__(self, dir_path='log_folder/'):
        self.log_path = os.path.join(dir_path, 'worker_log')
        self.open_events = set()

    def log_event_start(self, job_name, event):
        if event in self.open_events:
            return

        self.open_events.add(event)

        with open(self.log_path, 'a') as f:
            f.write(f"{job_name} {event.value} START {time.time()}\n")

    def log_event_end(self, job_name, event):
        if event not in self.open_events:
            return

        self.open_events.remove(event)

        with open(self.log_path, 'a') as f:
            f.write(f"{job_name} {event.value} END {time.time()}\n")