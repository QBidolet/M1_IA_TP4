import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Charger le dataset SPAM
df = pd.read_csv('../spam.csv',sep="\t")

# encoder les labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['text']
y = df['label']

# Découpage train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

"""
# Conversion en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Création des DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Définition du modèle
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        lx = self.linear(x)
        return torch.sigmoid(lx)

# Initialisation du modèle
input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Entraînement
n_epochs = 10
for epoch in range(n_epochs):

    model.train()
    
    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()

    # Calcul de l'accuracy sur l'ensemble d'entraînement
    with torch.no_grad():
        train_outputs = model(X_train_tensor)
        train_preds = (train_outputs >= 0.5).float()
        train_accuracy = (train_preds == y_train_tensor).float().mean()
        
        test_outputs = model(X_test_tensor)
        test_preds = (test_outputs >= 0.5).float()
        test_accuracy = (test_preds == y_test_tensor).float().mean()
    
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}, '
          f'Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}')

# Évaluation finale
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = (test_outputs >= 0.5).float()
    final_accuracy = (test_preds == y_test_tensor).float().mean()
    print(f'Final Test Accuracy: {final_accuracy:.4f}')

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Rapport de classification :")
print(classification_report(y_test, test_preds.numpy()))
"""
# =========================
# Arbre de décision (sklearn)
# =========================

# sklearn utilise numpy (pas les tensors)
X_train_np = X_train
X_test_np = X_test
y_train_np = y_train
y_test_np = y_test

"""
# Modèle
tree_model = DecisionTreeClassifier(random_state=42)

# Entraînement
tree_model.fit(X_train_np, y_train_np)

# Prédictions
y_pred_tree = tree_model.predict(X_test_np)

# Résultats
print("\nDecision Tree (SPAM)")
print("Accuracy:", accuracy_score(y_test_np, y_pred_tree))
print("Rapport de classification :")
print(classification_report(y_test_np, y_pred_tree))

# =========================
# SVM (sklearn)
# =========================

# Modèle SVM linéaire
svm_model = LinearSVC()

# Entraînement
svm_model.fit(X_train_np, y_train_np)

# Prédictions
y_pred_svm = svm_model.predict(X_test_np)

# Résultats
print("\nSVM")
print("Accuracy:", accuracy_score(y_test_np, y_pred_svm))
print("Rapport de classification :")
print(classification_report(y_test_np, y_pred_svm))

"""
# =========================
# Random Forest (sklearn)
# =========================

# Modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entraînement
rf_model.fit(X_train_np, y_train_np)

# Prédictions
y_pred_rf = rf_model.predict(X_test_np)

# Résultats
print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test_np, y_pred_rf))
print("Rapport de classification :")
print(classification_report(y_test_np, y_pred_rf))