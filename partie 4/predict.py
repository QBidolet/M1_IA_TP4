# VERSION MODIFIÉE DU CODE (ANTI-TRICHE)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

PREDICTIONS = {
    True: "Positif",
    False: "Négatif"
}


LR = 0.0001      # Learning rate => hyper paramètre
EPOCH = 10       # Nombre d'époque => hyper paramètre

#Chargement des données
USE_HF = False  # mettre True si datasets installé

if USE_HF:
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    X_train = dataset["train"]["text"][:5000]
    y_train = dataset["train"]["label"][:5000]
    X_test = dataset["test"]["text"][:5000]
    y_test = dataset["test"]["label"][:5000]
else:
    import pandas as pd
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
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# DataLoader (batch)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

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

mlp_model = MLP()

# ====== Loss (MODIFIÉ) ======
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=LR)


logistic_regression_model = LogisticRegression()

def train_logistic_regression():
    print("Entrainement régression logistique")
    logistic_regression_model.fit(X_train, y_train)

def get_logistic_regression_predict(avis):
    vec = vectorizer.transform([avis]).toarray()

    prediction = logistic_regression_model.predict(vec)[0]
    probabilite = logistic_regression_model.predict_proba(vec)[0][int(prediction)]

    vocab = vectorizer.vocabulary_
    reverse_vocab = {v:k for k,v in vocab.items()}
    idx = vec.argsort(axis=1)
    tfidf_max = idx[:,-3:]
    mots_importants = [[(reverse_vocab.get(item), vec[i, item])  for item in row] for i, row in enumerate(tfidf_max) ][0]

    return {
        "prediction": PREDICTIONS[prediction],
        "confiance": probabilite,
        "mots_importants": mots_importants
    }

# Entraînement
def train_mlp():
    mlp_model.train()
    for epoch in range(EPOCH):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = mlp_model(X_batch)
            loss = criterion(outputs, y_batch)
             # Compléter les étapes d'entrainement

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")



def get_mlp_predict(avis):
    mlp_model.eval()
    vec = vectorizer.transform([avis]).toarray()
    tensor = torch.tensor(vec, dtype=torch.float32)

    with torch.no_grad():
        output = mlp_model(tensor)
        probabilite = torch.sigmoid(output).item()
    
    vocab = vectorizer.vocabulary_
    reverse_vocab = {v:k for k,v in vocab.items()}
    idx = vec.argsort(axis=1)
    tfidf_max = idx[:,-3:]
    mots_importants = [[(reverse_vocab.get(item), vec[i, item])  for item in row] for i, row in enumerate(tfidf_max) ][0]

    prediction = probabilite > 0.5

    if prediction:
        confiance = probabilite
    else:
        confiance = 1 - probabilite

    return {
        "prediction": PREDICTIONS[prediction],
        "confiance": confiance,
        "mots_importants": mots_importants
    }
