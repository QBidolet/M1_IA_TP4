# VERSION MODIFIÉE DU CODE (ANTI-TRICHE)
import pandas as pd

import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer

#Chargement des données
USE_HF = False  # mettre True si datasets installé

if USE_HF:
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    X_train = dataset["train"]["text"][:5000]
    y_train = dataset["train"]["label"][:5000]
else:
    df1 = pd.read_csv('imdb_train.csv')
    X_train = df1['text']
    y_train = df1['label']
    df2 = pd.read_csv('imdb_test.csv')  
    X_test = df2['text']
    y_test = df2['label']


vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Conversion en tenseurs
X_train = torch.tensor(X_train, dtype=torch.float32)
# unsqueeze : Returns a new tensor with a dimension of size one inserted at the specified position.
# y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
# on enlève le unsqueeze pour le nouveau criterion
# sinon RuntimeError: 0D or 1D target tensor expected, multi-target not supported
y_train = torch.tensor(y_train, dtype=torch.long) # RuntimeError: expected scalar type Long but found Float

X_test = torch.tensor(X_test, dtype=torch.float32)
# unsqueeze : Returns a new tensor with a dimension of size one inserted at the specified position.
# y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.long) #idem

# DataLoader (batch)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Modèle MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # input = avant
        self.fc1 = nn.Linear(10000, 256)
        # batch normalization
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        # self.fc4 = nn.Linear(64, 1)
        self.fc4 = nn.Linear(64, 2) # deux au lieu de 1 pour le CrossEntropyLoss

        # A p=0.2 ça ne changeait presque rien, à 0.5 un peu plus d'impact.
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        x = torch.relu(self.fc2(x))
        # Pour tester car peu d'impact après la première
        # x = self.dropout(x)

        x = torch.relu(self.fc3(x))
        # Pour tester car peu d'impact après la première
        # x = self.dropout(x)

        x = self.fc4(x)  # BCEWithLogitsLoss => pas de sigmoid ici
        return x

model = MLP()

# ====== Loss (MODIFIÉ) ======
# criterion = fonction loss
# BCEWithLogitsLoss = Binary Cross Entropy + Sigmoid layer
# more numerically stable than using a plain Sigmoid followed by a Binary Cross Entropy Loss(output en proba)
# Binary Cross Entropy Loss
#
# This is a Loss Function that will measure how far our model’s predictions are from true labels. This is also used by Logistic Regression.
# Pytorch uses negative of rectified linear unit, that ensures that negative values are replaced with zeros. This is helpful for preventing large exponentiation of negative numbers, which can lead to numerical instability.
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

# optimizer == met à jour les poids des modèles à partir des gradients calculés. après avoir mesuré erreur
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
# Entraînement
def train():
    last_epoch_loss = 1

    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
             # Compléter les étapes d'entrainement
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # On calcule l'accuracy train comme au tp3 pour comparer directement les évolutions
            # preds = (torch.sigmoid(outputs) > 0.5).float()
            # correct_train += (preds == y_batch).sum().item()
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y_batch).sum().item()
            total_train += y_batch.size(0)

        epoch_loss /= len(train_loader)
        train_accuracy = correct_train / total_train

        # Accuracy test
        test_accuracy = compute_test_accuracy()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, "
              f"Test Acc: {test_accuracy:.4f}")

        if last_epoch_loss is not None:
            if epoch_loss > last_epoch_loss * 1.1: #10% d'erreur
                print("DIVERGENCE ?")

        # last_epoch_loss = epoch_loss
        # if abs(last_epoch_loss - epoch_loss) < 1e-6:
        #     print("CONVERGENCE")
        #     break



# Évaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # outputs = torch.sigmoid(model(X_batch))
            # preds = (outputs > 0.5).float()
            # correct += (preds == y_batch).sum().item()

            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()

            total += y_batch.size(0)
    print("Accuracy TEST :", correct / total)

# def compute_test_accuracy():
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             outputs = torch.sigmoid(model(X_batch))
#             preds = (outputs > 0.5).float()
#             correct += (preds == y_batch).sum().item()
#             total += y_batch.size(0)
#     model.train()  # Repasse en mode train après l'évaluation
#     return correct / total

def compute_test_accuracy():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)  # plus de sigmoid
            preds = torch.argmax(outputs, dim=1)  # plus de > 0.5
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    model.train()
    return correct / total

def evaluate_no_batching():
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)  # X_test est déjà un tensor float32

        # preds = (torch.sigmoid(outputs) > 0.5).float().squeeze()
        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == y_batch).sum().item()

        # y_true = y_test.squeeze()
        y_true = y_test

        print("MLP :")
        print(classification_report(y_true.numpy(), preds.numpy()))

# ====== Run ======
train()
start_time = time.time()
evaluate()
end_time = time.time()

execution_time = end_time - start_time
print("Execution Time BATCH: ", execution_time)
#
# start_time = time.time()
# evaluate_no_batching()
# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution Time: NO BATCH ", execution_time)