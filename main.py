from fastapi import FastAPI
from resnet_classify import classify_cat_image
from sentiment_classify import classify_sentiment

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/classify-image")
async def classify_image():
    return classify_cat_image()


@app.get("/classify-sentiment")
async def classify_text_sentiment():
    return classify_sentiment()
