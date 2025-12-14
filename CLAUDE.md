# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Run Commands
```bash
# Start FastAPI server
uvicorn main:app --reload
```

## Architecture
- `main.py` - FastAPI app with classification endpoints
- `resnet_classify.py` - ResNet-18 image classification function for FastAPI
- `sentiment_classify.py` - Text sentiment classification using cardiffnlp/twitter-roberta
- `masked_word_prediction.py` - Masked word prediction using BertForMaskedLM
- `ner_trainer.py` - NER training and inference using KLUE dataset and koELECTRA model

## API Endpoints
- `GET /` - Hello World
- `GET /classify-image` - Classify cat image using microsoft/resnet-18
- `GET /classify-sentiment` - Classify sentiment using cardiffnlp/twitter-roberta-base-sentiment-latest
- `GET /predict-masked-word` - Predict masked word using BertForMaskedLM
- `GET /train-test-ner` - Train NER model and test with Korean text (long-running ~4 minutes)
