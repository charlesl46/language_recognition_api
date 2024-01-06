import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import pickle
from rich.console import Console

C = Console()

df = pd.read_csv("dataset_collected.csv")
languages_to_keep = ["fr","en","es"]
df = df[df['language'].isin(languages_to_keep)]

print(f"Dataset's shape : {df.shape}")
langue_counts = df['language'].value_counts()
print("Language's repartition :\n", langue_counts)

nombre_de_langues = len(langue_counts)
print("Total number of languages :", nombre_de_langues)

corpus = df["sentence"].values
labels = df["language"].values

with C.status("Vectorization"):
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

param_grid = {'C': np.logspace(1,5,5), 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}

svm = SVC()

with C.status("Optimizing model"):
    grid_search = GridSearchCV(svm, param_grid, refit=True, verbose=3)
    grid_search.fit(X_train, y_train)

print("Best params :", grid_search.best_params_)
predictions = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Best model's accuracy : {accuracy}")
with open("svm_model.pkl","wb") as file:
    pickle.dump(grid_search,file)
C.log("SVM model serialized")

with open("w2vec_model.pkl","wb") as file:
    pickle.dump(model,file)
C.log("W2vec model serialized")

