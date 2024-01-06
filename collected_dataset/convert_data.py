import wikipedia
from peewee import Model, SqliteDatabase, CharField
import pandas as pd
from collect_data import Sentence

db = SqliteDatabase('database.db')

db.connect()

db.create_tables([Sentence], safe=True) 

wikipedia.set_lang("fr")

sentences_query = Sentence.select()

df = pd.DataFrame(list(sentences_query.dicts()))

df.to_csv("dataset_collected.csv", index=False)

db.close()
