import torch 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import csv
import numpy as np
import torch.nn as nn 
import torch.optim as opitm 
import torch.nn.functional as F  
import pandas as pd 
from sklearn.model_selection import train_test_split

#loading data to numpy matrix 
data_path = "./data.csv"
data_numpy = np.loadtxt(data_path, dtype=int, delimiter=",", skiprows=1)

seed = 1
# podział na wejscia
X = np.delete(data_numpy, -1, 1)

# wyjscia 
Y = data_numpy[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)


col_names = next(csv.reader(open(data_path), delimiter=','))
print(len(col_names))

#Numpy to pytorch tensor 
data_pytorch = torch.from_numpy(data_numpy)
# print(data_pytorch)
# print(data_pytorch.shape)

# podział danych: 70% training, 15% validating, 15% testing z prezentajci dr Ciskowskiego
 
# Od tego zacząłem z tutorialem
#Model class  
class Model (nn.Module) :
    # Warstwa wejściowa (w naszym przypadku mamy 14 wejsc) 
    # -> ukryta warstwa z ilomas neuronami H1 
    # ilość warstw czy tam neuronów możemy zmienić 
    # -> kolejna warstwa H2
    # -> wyjśice (8 wyjść po jednym na każdą ocenę)
    def __init__(self, in_features = 14, h1 =20, h2 = 15, out_features = 8):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x 

torch.manual_seed(seed)
#obiekt modelu
model = Model()

# Kryterium modelu, aby zmierzyć błąd 
criterion = nn.CrossEntropyLoss()

# Wybór optymizera (Adam) 
# jeśli błąd się nie zmniejsza obniżmay to= 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# Trenowanie modelu 
# ile epoch ( co to wgl jest)
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = model.forward(X_train) # dostaniemy przewidywanee rezultaty 

    # pomiar bledu
    loss = criterion(y_pred, Y_train)

    # przechowywnaie błędów
    losses.append(loss.detach().numpy())

    # drukuj co 10 
    # if i%10 == 0: 
    #     print(f'Epoch: {i} and loss: {loss}')
    
    # propagacja wsteczna 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#plot!
# plt.plot(range(epochs), losses)
# plt.xlabel('Epoch')
# plt.ylabel('loss/ error')

# plt.show()

# walidacja modelu na danych testowych 
with torch.no_grad(): # wysyłanie danych do modelu 
    y_eval = model.forward(X_test) # dajemy do modelu dane testowe 

    loss = criterion(y_eval, Y_test)

    
print(loss) # ?? wtf czemu tego tak dużo XD 
correct = 0

with torch.no_grad(): 
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        print(f'{i+1} | {str(y_val)} \t {Y_test[i]} \t {y_val.argmax().item()}')

        if y_val.argmax().item() == Y_test[i]:
            correct +=1

print(correct)
print('29')
 

