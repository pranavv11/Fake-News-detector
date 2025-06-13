import spacy
nlp = spacy.load("en_core_web_sm")

text = "Devastated by tragedy beyond words: PM Modi visits Air India crash site & meets survivors. 265 people have been killed in the London-bound"
doc = nlp(text)

keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN", "ADJ")]
print(keywords)