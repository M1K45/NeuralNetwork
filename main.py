import torch 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import csv
import numpy as np 

data_numpy = np.loadtxt("./data.csv", dtype=int, delimiter=",", skiprows=1)
print(data_numpy)
