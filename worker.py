from sklearn.metrics import roc_auc_score
import cv2
import argparse
import sys
import os
import zipfile
import tensorflow as tf
import numpy as np

class Worker:
    def __init__(self):
        return