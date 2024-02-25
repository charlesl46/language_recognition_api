import wikipedia
import random
import threading
from peewee import *

# Définir le modèle Peewee
db = SqliteDatabase('sentences.db')

class BaseModel(Model):
    class Meta:
        database = db

class Sentence(BaseModel):
    sentence = CharField()
    language = CharField()

# Connexion à la base de données
db.connect()

# Création de la table s'il elle n'existe pas encore
db.create_tables([Sentence])

# Fonction pour collecter les données à partir d'un article Wikipedia
def collect_data():
    global count
    try:
        while True:
            try:
                language = random.choice(["fr","en","es"])
                wikipedia.set_lang(language)
                article = wikipedia.random()
                page = wikipedia.page(article, preload=False)
                summary = page.summary
                sentences = summary.split(".")
                
                with db.atomic():
                    for sentence_text in sentences:
                        text = sentence_text.strip()
                        if len(text) > 1:
                            sentence, created = Sentence.get_or_create(sentence=text, language=language)
                            if created:
                                count += 1
                                print(f"{count} - {language}")
            except Exception:
                pass
    except KeyboardInterrupt:
        db.close()
        print("db closed")

threads = []
for _ in range(5): 
    t = threading.Thread(target=collect_data)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

