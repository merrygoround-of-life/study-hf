from fastapi import FastAPI
from resnet_classify import classify_cat_image
from sentiment_classify import classify_sentiment
from masked_word_prediction import predict_masked_word
from ner_trainer import test_ner_prediction_with_trained_model, load_ner_data_and_model
import asyncio
import concurrent.futures

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


@app.get("/predict-masked-word")
async def predict_masked_word_endpoint():
    return predict_masked_word()


@app.get("/train-test-ner")
async def train_and_test_ner():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(executor, test_ner_prediction_with_trained_model)
    return result
