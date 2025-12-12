from transformers import pipeline

def classify_sentiment():
    classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    
    texts = ["I love you", "I hate you", "I meet with you"]
    results = classifier(texts)
    
    classifications = []
    for text, result in zip(texts, results):
        classifications.append({
            "text": text,
            "label": result["label"],
            "score": float(result["score"])
        })
    
    return {"classifications": classifications}