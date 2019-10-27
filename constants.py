from enum import Enum

class Command(Enum):
    TRAIN = "TRAIN"
    START_PS = "START_PS"
    STOP_PS = "STOP_PS"
    POLL = "POLL"
    RESET = "RESET"

class Status(Enum):
    FREE = "FREE"
    TRAINING = "TRAINING"