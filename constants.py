from enum import Enum
import time
import os

class Command(Enum):
    TRAIN = "TRAIN"
    VALIDATE = "VALIDATE"
    TEST = "TEST"
    START_PS = "START_PS"
    STOP_PS = "STOP_PS"
    POLL = "POLL"
    RESET = "RESET"
    CLEAN = "CLEAN"

class Status(Enum):
    FREE = "FREE"
    BUSY = "BUSY"

class Event(Enum):
    DOWNLOAD = "DOWNLOAD"
    BUILD = "BUILD"
    TRAIN = "TRAIN"
    SAVE = "SAVE"
    RESTORE = "RESTORE"
    VAL_ACC = "VAL_ACC"
    VAL_LOSS = "VAL_LOSS"

class Logger:
    def __init__(self, job_name, dir_path='log_folder/'):
        self.log_path = os.path.join(dir_path, 'worker_log')
        self.open_events = set()
        self.job_name = job_name

    def log_event_start(self, event):
        if event in self.open_events:
            return

        self.open_events.add(event)

        with open(self.log_path, 'a') as f:
            f.write(f"{self.job_name} {event.value} START {time.time()}\n")

    def log_event_end(self, event):
        if event not in self.open_events:
            return

        self.open_events.remove(event)

        with open(self.log_path, 'a') as f:
            f.write(f"{self.job_name} {event.value} END {time.time()}\n")

    def log_val_acc(self, acc):
        with open(self.log_path, 'a') as f:
            f.write(f"{self.job_name} {Event.VAL_ACC.value} {acc}\n")

    def log_val_loss(self, loss):
        with open(self.log_path, 'a') as f:
            f.write(f"{self.job_name} {Event.VAL_LOSS.value} {loss}\n")