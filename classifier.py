import os
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast


@torch.no_grad()
def get_prediction(reviews, save_model=False):
    labels = ['neutral', 'positive', 'negative']
    tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')

    if True in list(map(lambda file: file.endswith('.pt'), os.listdir())):
        model = torch.load('model.pt')
        model.eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)

    if save_model:
        torch.save(model, 'model.pt')

    predictions = []
    for review in reviews:
        inputs = tokenizer(review, max_length=512, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs)
        predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(predicted, dim=1).numpy()
        predictions.append(labels[int(predicted)])

    return predictions