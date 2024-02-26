"""
.. include:: README.md
"""

from flask import Flask, request, jsonify
import pickle
from gensim.models import Word2Vec
from sklearn.svm import SVC
import numpy as np
import json

app = Flask(__name__)

with open("collected_dataset/svm_model.pkl","rb") as file:
    svm_model : SVC = pickle.load(file)
print("SVM loaded")

with open("collected_dataset/w2vec_model.pkl","rb") as file:
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

def detect(sentence : str):
    print(f"Trying to detect language for sentence {sentence}")
    vector = to_vec(sentence)
    return svm_model.predict(vector.reshape(1,-1))[0]

@app.route('/detect-language', methods=['POST'])
def detect_language():
    try:
        data = json.loads(request.get_json())
        sentence = data.get("sentence")
        detected_language = detect(sentence)
        result = {'detected_language': detected_language}
        return jsonify(json.dumps(result)), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)