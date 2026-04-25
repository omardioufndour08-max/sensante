import os
import pandas as pd
import numpy as np
import joblib

# 🔹 Configuration affichage graphique
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Création des dossiers
os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 🔹 Chargement des données
df = pd.read_csv("data/patients_dakar.csv")

print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"\nColonnes : {list(df.columns)}")
print(f"\nDiagnostics :\n{df['diagnostic'].value_counts()}")

# 🔹 Encodage
from sklearn.preprocessing import LabelEncoder

le_sexe = LabelEncoder()
le_region = LabelEncoder()

df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

# 🔹 Features
feature_cols = [
    'age', 'sexe_encoded', 'temperature', 'tension_sys',
    'toux', 'fatigue', 'maux_tete', 'frissons', 'nausee',
    'region_encoded'
]

X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")

# 🔹 Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

# 🔹 Modèle
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Modele entraine !")
print(f"Nombre d'arbres : {model.n_estimators}")
print(f"Classes : {list(model.classes_)}")

# 🔹 Prédictions
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Vrai diagnostic': y_test.values[:10],
    'Prediction': y_pred[:10]
})
print(comparison)

# 🔹 Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

# 🔹 Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("Matrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# 🔹 Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel('Prediction du modele')
plt.ylabel('Vrai diagnostic')
plt.title('Matrice de confusion - SenSante')

plt.tight_layout()

# 🔹 Sauvegarde image
plt.savefig('figures/confusion_matrix.png', dpi=150)

# 🔹 Affichage non bloquant
plt.show(block=False)
plt.pause(5)
plt.close()

# 🔹 Sauvegarde modèle et encodeurs
joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

print("Modele et encodeurs sauvegardes.")

# 🔹 Chargement pour test
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
feature_cols_loaded = joblib.load("models/feature_cols.pkl")

# 🔹 Nouveau patient
nouveau_patient = {
    'age': 28,
    'sexe': 'F',
    'temperature': 39.5,
    'tension_sys': 110,
    'toux': True,
    'fatigue': True,
    'maux_tete': True,
    'frissons': True,
    'nausee': False,
    'region': 'Dakar'
}

# 🔹 Encodage
sexe_enc = le_sexe_loaded.transform([nouveau_patient['sexe']])[0]
region_enc = le_region_loaded.transform([nouveau_patient['region']])[0]

# 🔹 Création DataFrame (CORRECTION IMPORTANTE)
features_dict = {
    'age': nouveau_patient['age'],
    'sexe_encoded': sexe_enc,
    'temperature': nouveau_patient['temperature'],
    'tension_sys': nouveau_patient['tension_sys'],
    'toux': int(nouveau_patient['toux']),
    'fatigue': int(nouveau_patient['fatigue']),
    'maux_tete': int(nouveau_patient['maux_tete']),
    'frissons': int(nouveau_patient['frissons']),
    'nausee': int(nouveau_patient['nausee']),
    'region_encoded': region_enc
}

features_df = pd.DataFrame([features_dict])

# 🔹 Assurer ordre correct
features_df = features_df[feature_cols_loaded]

# 🔹 Prédiction
diagnostic = model_loaded.predict(features_df)[0]
probas = model_loaded.predict_proba(features_df)[0]

print("\n--- Resultat du pre-diagnostic ---")
print(f"Patient : {nouveau_patient['sexe']}, {nouveau_patient['age']} ans")
print(f"Diagnostic : {diagnostic}")

for classe, proba in zip(model_loaded.classes_, probas):
    bar = '#' * int(proba * 30)
    print(f"{classe:10s} : {proba:.1%} {bar}")