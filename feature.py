import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split

# TODO: 
# problem przeuczenia 
# moze dropout? 
# albo inna normalizacja 
# 


#plik csv
data_path = "data.csv"

with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    col_names = next(reader)
print("Columns:", col_names)

#pomiń pierwsy
data_numpy = np.loadtxt(data_path, dtype=int, delimiter=",", skiprows=1)
print("Data shape:", data_numpy.shape)

#cechy i etykiety - tutaj dzielimy dane na wejścia i wyjścia 
X = data_numpy[:, :-1]
y = data_numpy[:, -1]  #ostatnia kolumna to GRADE

#konwersja do tensora PyTorch
X_t = torch.from_numpy(X).float()
y_t = torch.from_numpy(y).long()

#podział na zbiór treningowy i walidacyjny - zbiór treningowy ma 116 elementów
X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_t, y_t, test_size=0.3, random_state=42, shuffle=True)
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
input_size = X_train_t.shape[1]       # liczba wejść w warstwie wejściowej - czynników
num_classes = len(torch.unique(y_t))  # liczba unikalnych klas w etykiecie
print("Number of classes:", num_classes)

#-------------------------------------------------------------------------
#konfiguracja modelu
#tzreba dodać regularyzacje np dropout bo jest przeuczenie i jakos pobawić sie z normalizacją bo MSE nie takie

epochs = 1000
learning_rate = 0.001
momentum = 0.9
early_stopping_threshold = 0.01
hidden_dim = 2

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3) 
        # self.fc2 = nn.Linear(hidden_dim, 7)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

model = SimpleMLP(input_size, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=1e-4)



#funkcje pomocnicze
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def calculate_mse(outputs, targets, num_classes):
    probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
    one_hot_targets = one_hot_encode(targets.cpu().numpy(), num_classes)
    mse_val = np.mean((probs - one_hot_targets) ** 2)
    return mse_val

def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)

def get_weights(model):
    w1 = model.fc1.weight.detach().cpu().numpy()
    w2 = model.fc2.weight.detach().cpu().numpy()
    return w1, w2

# -------------------------------------------------------------------------
#trening

train_mse_list = []
val_mse_list = []
train_acc_list = []
val_acc_list = []
w1_list = []
w2_list = []

for epoch in range(epochs):
    model.train()
    total_samples = 0
    total_correct = 0
    total_mse_train = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_samples += batch_X.size(0)
            total_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            batch_mse = calculate_mse(outputs, batch_y, num_classes)
            total_mse_train += batch_mse * batch_X.size(0)

    train_acc = total_correct / total_samples
    train_mse = total_mse_train / total_samples

    #walidacja
    model.eval()
    val_total_samples = 0
    val_total_correct = 0
    val_total_mse = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_total_samples += batch_X.size(0)
            val_total_correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            batch_mse = calculate_mse(outputs, batch_y, num_classes)
            val_total_mse += batch_mse * batch_X.size(0)

    val_acc = val_total_correct / val_total_samples
    val_mse = val_total_mse / val_total_samples
    train_mse_list.append(train_mse)
    val_mse_list.append(val_mse)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    if epoch % 50 == 0:
        w1, w2 = get_weights(model)
        w1_list.append(w1)
        w2_list.append(w2)

    if epoch % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] - Train MSE: {train_mse:.4f}, Train Acc: {train_acc:.4f}, Val MSE: {val_mse:.4f}, Val Acc: {val_acc:.4f}")

    if val_mse < early_stopping_threshold:
        print("wczesne zakończenie uczenia - osiągnięto zadany próg bledu MSE.")
        break

#-------------------------------------------------------------------------
#wizualizacje

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_mse_list, label='Train MSE')
plt.plot(val_mse_list, label='Val MSE')
plt.xlabel('Epoka')
plt.ylabel('MSE')
plt.title('błąd MSE w trakcie uczenia')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot([1 - acc for acc in train_acc_list], label='Train Error Rate')
plt.plot([1 - acc for acc in val_acc_list], label='Val Error Rate')
plt.xlabel('epoka')
plt.ylabel('błąd klasyfikacji')
plt.title('błąd klasyfikacji w trakcie uczenia')
plt.legend()
plt.tight_layout()
plt.show()

#wizualizacja wag (co 50 epok)
# for i, (w1_snapshot, w2_snapshot) in enumerate(zip(w1_list, w2_list)):
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(w1_snapshot, aspect='auto', cmap='bwr')
#     plt.colorbar()
#     plt.title(f'wagi warstwy 1 - po {i * 50} epokach')
#     plt.subplot(1, 2, 2)
#     plt.imshow(w2_snapshot, aspect='auto', cmap='bwr')
#     plt.colorbar()
#     plt.title(f'wagi warstwy 2 - po {i * 50} epokach')
#     plt.tight_layout()
#     plt.show()
