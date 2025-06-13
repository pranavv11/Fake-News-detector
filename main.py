from fastapi import FastAPI, Request, Form
import os, requests, uuid
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import cv2
import numpy as np
import spacy

model = spacy.load("en_core_web_sm")

load_dotenv()
app = FastAPI()

NEWS_API_KEY=os.getenv('NEWS_API_KEY')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

media_dir = "Media"
os.makedirs(media_dir, exist_ok=True)

def verify_news(keywords):
    query = ' '.join(keywords)
    url = "https://newsdata.io/api/1/news"

    params = {
        'apikey' : NEWS_API_KEY,
        'q' : query,
        'language' : 'en',
        'country' : 'in,us,gb',
        'category' : 'top'
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("API Call fail")
        return "Uncertain", []

    data = response.json()
    articles = data.get("results", [])

    if not articles: 
        return "Possibly Fake", []
    
    sources = [(article["title"], article["link"]) for article in articles]

    return "Likely Real", sources


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

        tokens = model(text)
        keywords = [token.text for token in tokens if token.pos_ in ("NOUN", "PROPN", "ADJ")]
        print(f"Keywords of text: {keywords}")

        return PlainTextResponse("Image Received")
    
    return PlainTextResponse("No valid image detected")