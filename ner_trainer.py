from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import numpy as np
import torch

def load_ner_data_and_model():
    dataset = load_dataset("klue", "ner")
    labels = dataset["train"].features["ner_tags"].feature.names

    model_name = "Leo97/koELECTRA-small-v3-modu-ner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                            num_labels=len(labels),
                                                            ignore_mismatched_sizes=True)
    
    return dataset, tokenizer, model, labels

def preprocessed_dataset(dataset, tokenizer):
    def preprocess_ner_data(example):
        sentence = "".join(example["tokens"]).replace("\xa0", " ")
        encoded = tokenizer(sentence,
                            return_offsets_mapping=True,
                            add_special_tokens=False,
                            padding=False,
                            truncation=False)

        labels = []
        for offset in encoded.offset_mapping:
            if offset[0] == offset[1]:
                labels.append(-100)
            else:
                labels.append(example["ner_tags"][offset[0]])
        encoded["labels"] = labels
        return encoded

    return dataset.map(lambda example: preprocess_ner_data(example),
                       batched=False,
                       remove_columns=dataset["train"].column_names)


def train_ner_model():
    dataset, tokenizer, model, labels = load_ner_data_and_model()
    
    tokenized_dataset = preprocessed_dataset(dataset, tokenizer)
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=".token-classification",
        eval_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_train_epochs=5,
        seed=42,
        report_to=[])
    
    # Trainer
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=data_collator,
                      processing_class=tokenizer)
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    return trainer, eval_results, labels

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def predict_ner(text, trainer, labels):
    model = trainer.model
    tokenizer = trainer.processing_class
    device = get_device()
    model.to(device)

    with torch.no_grad():
        tokenized = tokenizer(text,
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=512,
                              add_special_tokens=False)
        logits = model(**tokenized.to(device)).logits.cpu()

    predictions = logits.argmax(dim=-1)[0].tolist()
    tokens = tokenizer.tokenize(text)

    result = []
    for token, label in zip(tokens, [labels[i] for i in predictions]):
        result.append({"token": token, "label": label})
    
    return result

def test_ner_prediction_with_trained_model(text="2025년 12월 14일 나는 한국에서 허깅페이스 트랜스포머로 모델 학습 및 평가를 공부했다."):
    trainer, eval_results, labels = train_ner_model()
    predictions = predict_ner(text, trainer, labels)
    
    return {
        "input_text": text,
        "predictions": predictions,
        "eval_results": eval_results
    }