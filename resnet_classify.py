from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
from PIL import Image

def classify_cat_image():
    # Load the ResNet-18 model and processor
    model_name = "microsoft/resnet-18"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    
    # Load cats-image dataset from HuggingFace
    print("Loading cats-image dataset...")
    dataset = load_dataset("huggingface/cats-image")
    
    # Get the first test image
    test_image = dataset["test"]["image"][0]
    
    # Preprocess the image
    inputs = processor(test_image, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # Get the predicted class
    predicted_class_idx = predictions.argmax().item()
    confidence = predictions.max().item()
    
    # Get class label
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return {
        "predicted_class": predicted_label,
        "confidence": float(confidence),
        "image_size": test_image.size,
        "image_mode": test_image.mode
    }
