import wikipedia
from peewee import Model, SqliteDatabase, CharField
from collect_data import Sentence

if __name__ == "__main__":

    db = SqliteDatabase('collected_dataset/database.db')

    db.connect()
    db.create_tables([Sentence], safe=True) 

    wikipedia.set_lang("fr")

    sentences = Sentence.select()
    print(f"English sentences {Sentence.select().where(Sentence.language == 'en').count()}")
    print(f"French sentences {Sentence.select().where(Sentence.language == 'fr').count()}")
    print(f"Spanish sentences {Sentence.select().where(Sentence.language == 'es').count()}")


    """ for st in sentences:
        print(f"[{st.language}] {st.sentence}") """

    db.close()