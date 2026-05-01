# VERSION MODIFIÉE DU CODE (ANTI-TRICHE)

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

LR_TEST = [0.01, 0.001, 0.0001]
EPOCH_TEST = [2, 5, 7, 10]

LR = 0.0001      # Learning rate => hyper paramètre
EPOCH = 10       # Nombre d'époque => hyper paramètre

#Chargement des données
USE_HF = True  # mettre True si datasets installé

if USE_HF:
    from datasets import load_dataset

    dataset = load_dataset("imdb")
    # Mélanger avant de prendre 5000 (le dataset HF est trié par classe, on a que des 0 sinon)
    train_shuffled = dataset["train"].shuffle(seed=12)
    test_shuffled = dataset["test"].shuffle(seed=12)
    unsup_shuffled = dataset["unsupervised"].shuffle(seed=12)

    X_train = train_shuffled["text"][:5000]
    y_train = train_shuffled["label"][:5000]
    X_test = test_shuffled["text"][:5000]
    y_test = test_shuffled["label"][:5000]
    X_unsupervised = unsup_shuffled["text"][:5000]
else:
    import pandas as pd
    df1 = pd.read_csv('../imdb_train.csv')
    X_train = df1['text']
    y_train = df1['label']

    df2 = pd.read_csv('../imdb_test.csv')
    X_test = df2['text']
    y_test = df2['label']

    df3 = pd.read_csv('imdb_unsupervised.csv')  # Chargement des données non labelisés
    X_unsupervised = df3['text']


vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
X_unsupervised = vectorizer.transform(X_unsupervised).toarray()

# Conversion en tenseurs
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

X_unsupervised = torch.tensor(X_unsupervised, dtype=torch.float32)

# labellisation avec kmeans, on fixe le random state pour la reproductibilité
kmeans = KMeans(n_clusters=2, random_state=12, n_init=10)
y_unsupervised = kmeans.fit_predict(X_unsupervised)
y_unsupervised = torch.tensor(y_unsupervised, dtype=torch.float32).unsqueeze(1)

# DataLoader (batch)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

unsupervised_dataset = TensorDataset(X_unsupervised, y_unsupervised)
combined_dataset = ConcatDataset([train_dataset, unsupervised_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
unsupervised_loader = DataLoader(unsupervised_dataset, batch_size=64)

# Modèle MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10000, 256)    # Le in ici est l'input layer
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.dr1 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)         # Le out ici est l'output layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # BCEWithLogitsLoss => pas de sigmoid ici
        return x

model = MLP()

# ====== Loss (MODIFIÉ) ======
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Entraînement
def train():
    model.train()
    for epoch in range(EPOCH):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
             # Compléter les étapes d'entrainement

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Entraînement
def train_with_unsupervised():
    model.train()
    for epoch in range(EPOCH):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
             # Compléter les étapes d'entrainement

            loss.backward()
            optimizer.step()

        for X_batch, y_batch in unsupervised_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
             # Compléter les étapes d'entrainement

            loss.backward()
            optimizer.step()


        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Évaluation
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = torch.sigmoid(model(X_batch))
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    print("Accuracy:", correct / total)

# ====== Run ======
# for LR in LR_TEST:
#     for EPOCH in EPOCH_TEST:
#         print(f'Test avec {LR} LR et {EPOCH} epoque')

# train()
train_with_unsupervised()
evaluate()
