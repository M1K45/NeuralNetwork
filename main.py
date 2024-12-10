import torch 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import csv
import numpy as np 

#loading data to numpy matrix 
data_path = "./data.csv"
data_numpy = np.loadtxt(data_path, dtype=int, delimiter=",", skiprows=1)
# print(data_numpy)

col_names = next(csv.reader(open(data_path), delimiter=','))
print (col_names)

#Numpy to pytorch tensor 
data_pytorch = torch.from_numpy(data_numpy)
print(data_pytorch)
print(data_pytorch.shape)