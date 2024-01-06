import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

with open("svm_model.pkl","rb") as file:
    svm_model : SVC = pickle.load(file)
print("SVM loaded")

with open("w2vec_model.pkl","rb") as file:
    w2vec_model : Word2Vec = pickle.load(file)
print("W2vec loaded")

def to_vec(sentence : str):
    mots = sentence.split()
    vecteurs_mots = [w2vec_model.wv[mot] for mot in mots if mot in w2vec_model.wv]

    if vecteurs_mots:
        vecteur_phrase = np.mean(vecteurs_mots, axis=0)
    else:
        vecteur_phrase = np.zeros(w2vec_model.vector_size)
    return vecteur_phrase

sentence = ""
vector = to_vec(sentence)
print(svm_model.predict(vector.reshape(1,-1)))

