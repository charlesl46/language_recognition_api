import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stopwords = {"fr" : set(stopwords.words('french')),"en" : set(stopwords.words('english')),"es" : set(stopwords.words('spanish'))} 

def preprocess(text : str,language : str):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stopwords.get(language)]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return str(" ".join(lemmatized_tokens))

if __name__ == "__main__":
    ex = "Jacopo Filippo Foresti est un moine augustin, théologien, chroniqueur et historien italien, plus connu sous le nom de Jacques-Philippe de Bergame"
    print(preprocess(ex,"fr"))