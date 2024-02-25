import wikipedia
from peewee import Model, SqliteDatabase, CharField
import pandas as pd
from collect_data import Sentence
from preprocess import preprocess
from rich.progress import track

db = SqliteDatabase('database.db')

db.connect()
db.create_tables([Sentence], safe=True) 
wikipedia.set_lang("fr")

sentences_query = Sentence.select()


dicts = sentences_query.dicts()



ids = [dico.get("id") for dico in dicts]
languages = [dico.get("language") for dico in dicts]

sentences = [preprocess(dico.get("sentence"),lang) for dico,lang in track(zip(dicts,languages))]

df = pd.DataFrame()
df["id"] = pd.Series(ids)
df["sentence"] = pd.Series(sentences)
df["language"] = pd.Series(languages)
df.to_csv("dataset_collected.csv", index=False)
db.close()
