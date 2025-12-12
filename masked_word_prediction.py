from transformers import BertTokenizer, BertForMaskedLM
import torch

def predict_masked_word():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    masked_sentence = f"I {tokenizer.mask_token} learning about transformers."
    
    tokens = tokenizer.encode(masked_sentence, return_tensors='pt')
    mask_token_index = torch.where(tokens == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        outputs = model(tokens)
        predictions = outputs.logits
    
    mask_predictions = predictions[0, mask_token_index, :]
    top10_tokens = torch.topk(mask_predictions, 10, dim=1)
    softmax_predictions = torch.softmax(mask_predictions, dim=1)
    
    results = []
    for token_tensor in top10_tokens.indices[0]:
        token_id = token_tensor.item()
        confidence = softmax_predictions[0][token_id].item()
        predicted_token = tokenizer.decode([token_id])
        predicted_sentence = masked_sentence.replace(tokenizer.mask_token, predicted_token)
        
        results.append({
            'sentence': predicted_sentence,
            'predicted_word': predicted_token,
            'confidence': confidence
        })
    
    return results