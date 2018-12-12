import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import time

def main

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',help='path to train json')

parser.add_argument('test_json', metavar='TEST',help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,help='task id to use.')

with open(args.train_json, 'r') as outfile:        
	train_list = json.load(outfile)
with open(args.test_json, 'r') as outfile:       
	val_list = json.load(outfile)
