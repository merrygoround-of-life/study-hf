from fastapi import FastAPI
from resnet_classify import classify_cat_image

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.get("/classify")
async def classify_image():
    return classify_cat_image()
