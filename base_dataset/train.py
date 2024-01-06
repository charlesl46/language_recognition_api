import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pickle

df = pd.read_csv("dataset.csv")
languages_to_keep = ["French","English","Spanish"]
df = df[df['language'].isin(languages_to_keep)]

print(f"Taille du dataset : {df.shape}")
first_part = df
print(f"Taille des données sélectionnées : {first_part.shape}")
langue_counts = df['language'].value_counts()
print("Répartition des langues :\n", langue_counts)

nombre_de_langues = len(langue_counts)
print("Nombre total de langues :", nombre_de_langues)

corpus = first_part["Text"].values
labels = first_part["language"].values

model = Word2Vec(corpus.tolist(), vector_size=100, window=5, min_count=1, workers=4)

vecteurs_phrases = []
for phrase in corpus.tolist():
    mots = phrase.split()
    vecteurs_mots = [model.wv[mot] for mot in mots if mot in model.wv]

    if vecteurs_mots:
        vecteur_phrase = np.mean(vecteurs_mots, axis=0)
    else:
        vecteur_phrase = np.zeros(model.vector_size)
    vecteurs_phrases.append(vecteur_phrase)

X_train, X_test, y_train, y_test = train_test_split(vecteurs_phrases, labels, test_size=0.2)

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}

svm = SVC()

grid_search = GridSearchCV(svm, param_grid, refit=True, verbose=3)
grid_search.fit(X_train, y_train)
print("Meilleurs paramètres :", grid_search.best_params_)
predictions = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Précision du modèle optimisé : {accuracy}")
with open("svm_model.pkl","wb") as file:
    pickle.dump(grid_search,file)

with open("w2vec_model.pkl","wb") as file:
    pickle.dump(model,file)
