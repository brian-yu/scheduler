from enum import Enum

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