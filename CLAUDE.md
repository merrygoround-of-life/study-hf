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
- `main.py` - FastAPI app with endpoints
- `resnet_classify.py` - ResNet-18 image classification function for FastAPI
- `sentiment_classify.py` - Sentiment classification using twitter-roberta model
- `masked_word_prediction.py` - Masked word prediction using BertForMaskedLM

## API Endpoints
- `GET /` - Hello World
- `GET /classify-image` - Classify cat image using microsoft/resnet-18
- `GET /classify-sentiment` - Sentiment classification using twitter-roberta
- `GET /predict-masked-word` - Predict masked word using BertForMaskedLM