# from transformers import AutoTokenizer, AutoModel

# # load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
# model = AutoModel.from_pretrained('allenai/specter')

# papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
#           {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# # concatenate title and abstract
# title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# # preprocess the input
# inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
# result = model(**inputs)
# # take the first token in the batch as the embedding
# embeddings = result.last_hidden_state[:, 0, :]
# print(embeddings)

import json
import pandas as pd
import torch



class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, ids_path, tokenizer):
        df = pd.read_csv(ids_path)
        with open(metadata_path, encoding='utf-8') as f:
            metadata = json.load(f)

        def get_title_and_abs(pid):
            paper = metadata[pid]
            title, abstract = paper['title'], paper['abstract']
            return title + " " + tokenizer.sep_token + " " + (abstract or '')
        
        self.X = df['pid'].apply(get_title_and_abs).tolist()
        self.encodings = tokenizer(self.X, padding=True, truncation=True, max_length=512)
        self.labels = df['class_label'].tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
    def num_labels(self):
        return len(set(self.labels))


model_name = 'allenai/specter'

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained(model_name)
print('tokenizer loaded')

train_ds = ClassificationDataset('./data/metadata.json', './data/train.csv', tokenizer)
val_ds = ClassificationDataset('./data/metadata.json', './data/val.csv', tokenizer)
test_ds = ClassificationDataset('./data/metadata.json', './data/test.csv', tokenizer)
print('ds loaded')

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_ds.num_labels())
model = model.to('cuda')
print('model loaded')

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    return {
        "acc": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weight": f1_score(labels, predictions, average="weighted")
    }



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=10,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_ds,         # training dataset
    eval_dataset=val_ds,             # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.train()

preds = trainer.predict(test_ds)

y_pred = [np.argmax(array, axis=0) for array in preds.predictions] 
print(classification_report(test_ds.labels, y_pred))

# load model and tokenizer

# result = model(**inputs)
print('done')
# preprocess the input

# take the first token in the batch as the embedding
# embeddings = result.last_hidden_state[:, 0, :]
# print(embeddings, embeddings.requires_grad)


