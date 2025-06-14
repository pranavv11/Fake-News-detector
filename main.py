from fastapi import FastAPI, Request, Form
import os, requests, uuid
from fastapi.responses import PlainTextResponse
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import cv2
import numpy as np
import spacy

model = spacy.load("en_core_web_sm")
sim_model = SentenceTransformer('all-MiniLM-L6-V2')

load_dotenv()
app = FastAPI()

NEWS_API_KEY=os.getenv('NEWS_API_KEY')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

media_dir = "Media"
os.makedirs(media_dir, exist_ok=True)

def verify_news(keywords):
    query = ' '.join(keywords)
    print(query)
    url = "https://newsdata.io/api/1/news?apikey=pub_e94d0b515c594cc19f1ceec80549849b&q=test"

    params = {
        'apikey' : NEWS_API_KEY,
        'q' : query,
        'language' : 'en',
        # 'country' : 'in,us,gb',
        # 'category' : 'top'
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("API Call fail", response.status_code)
        return "Uncertain", []

    data = response.json()
    articles = data.get("results", [])

    if not articles: 
        return "Possibly Fake", []
    
    sources = [(article["title"], article["link"]) for article in articles]

    return "Likely Real", sources

def compute_similarity(text, titles):
    text_embedding = sim_model.encode(text, convert_to_tensor=True)
    title_embedding = sim_model.encode(titles, convert_to_tensor=True)

    cosine_score = util.cos_sim(text_embedding, title_embedding)[0]
    max_score = float(cosine_score.max())
    best_title = titles[cosine_score.argmax()]

    return max_score, best_title

def send_reply(to, message):
    url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json"
    data = {
        "From": TWILIO_PHONE_NUMBER,
        "To" : to, 
        "Body" : message
    }

    response = requests.posts(url, data=data, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
    print(f"Response status: {response.status_code}")

@app.post("/webhook")
async def receive_image(request: Request):
    form = await request.form()
    media_url = form.get("MediaUrl0")
    media_type = form.get("MediaContentType0")
    sender = form.get("From")

    if media_url and "image" in media_type:
        ext = media_type.split("/")[-1]
        filename = f"{uuid.uuid4()}.{ext}"
        path = os.path.join(media_dir, filename)
        res = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))

        with open(path, 'wb') as f:
            f.write(res.content)
        
        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        print(f"Text from image: {text}")

        # tokens = model(text)
        # keywords = [token.text for token in tokens if token.pos_ in ("NOUN", "PROPN", "ADJ")]
        # print(f"Keywords of text: {keywords}")
        keywords = []
        doc = model(text)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'GPE', 'PERSON', 'EVENT']:
                keywords.append(ent.text)
        for token in doc:
            if (token.pos_ in ['PROPN', 'NOUN'] 
                and not token.is_stop 
                and len(token.text) > 2  # Filter out short words
                and token.text.lower() not in [kw.lower() for kw in keywords]):
                    keywords.append(token.text)
        print(keywords)

        verdict, sources = verify_news(keywords)
        if sources:
            titles = [title for title, link in sources]
            score, matched_title = compute_similarity(text, titles)

            if score > 0.3:
                final_verdict = "Most likely real"
            else: 
                final_verdict = "Possibly Fake"
        else: 
            final_verdict = "Possibly Fake"

        return PlainTextResponse(f"News Check: {final_verdict}")
    
    return PlainTextResponse("No valid image detected")