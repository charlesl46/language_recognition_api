import wikipedia,random
from peewee import Model, SqliteDatabase, CharField
from rich.console import Console
import warnings

warnings.filterwarnings("ignore")

db = SqliteDatabase('database.db')

class Sentence(Model):
    sentence = CharField()
    language = CharField()

    class Meta:
        database = db

if __name__=="__main__":
    db.connect()

    db.create_tables([Sentence])


    count = Sentence.select().count()
    C = Console()
    with C.status("Collecting data"):
        try:
            while True:
                try:
                    language = random.choice(["fr","en","es"])
                    wikipedia.set_lang(language)
                    article = wikipedia.random()
                    page = wikipedia.page(article, preload=False)
                    summary : str = page.summary
                    sentences = summary.split(".")
                    
                    with db.atomic():
                        for sentence_text in sentences:
                            text = sentence_text.strip()
                            if len(text) > 1:
                                sentence, created = Sentence.get_or_create(sentence=text, language=language)
                                if created:
                                    count += 1
                                    C.log(f"{count} - {language}")
                except Exception:
                    pass
        except KeyboardInterrupt:
            db.close()
            print("db closed")


