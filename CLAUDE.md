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
- `main.py` - FastAPI app with endpoints including `/classify`
- `resnet_classify.py` - ResNet-18 image classification function for FastAPI

## API Endpoints
- `GET /` - Hello World
- `GET /classify` - Classify cat image using microsoft/resnet-18