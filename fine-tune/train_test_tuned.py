import json
import pandas as pd
import torch
import sys
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, ids_path, tokenizer):
        df = pd.read_csv(ids_path)
        with open(metadata_path, encoding='utf-8') as f:
            metadata = json.load(f)

        metadata_ids = metadata.keys()
        valid_ids = df['pid'].apply(lambda pid: pid in metadata_ids)
        df = df[valid_ids]
        print("missing papers for ids", ids_path, "=", sum(~valid_ids))
        print("valid papers for ids", ids_path, "=", sum(valid_ids))

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
if '--scibert' in sys.argv:
    model_name = 'allenai/scibert_scivocab_uncased'
print('using model', model_name)


metadata_folder = './data/cls-metadata.json'

if '--mag' in sys.argv:
    ids_folder = 'mag'
elif '--mesh' in sys.argv:
    ids_folder = 'mesh'
else:
    raise ValueError("select either --mag or --mesh datasets")


tokenizer = AutoTokenizer.from_pretrained(model_name)
print('tokenizer loaded')


train_ds = ClassificationDataset(metadata_folder, f'./data/{ids_folder}/train.csv', tokenizer)
val_ds = ClassificationDataset(metadata_folder, f'./data/{ids_folder}/val.csv', tokenizer)
test_ds = ClassificationDataset(metadata_folder, f'./data/{ids_folder}/test.csv', tokenizer)
print('ds loaded')

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=train_ds.num_labels())
model = model.to('cuda')
print('model loaded')



def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    return {
        "acc": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weight": f1_score(labels, predictions, average="weighted")
    }



training_args = TrainingArguments(
    output_dir='./results',         
    num_train_epochs=1,             
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=16,  
    warmup_steps=10,                
    weight_decay=0.1,               
    logging_dir='./logs',           
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    eval_steps=100,
    save_steps=100,
    load_best_model_at_end=True,
    learning_rate=5e-5,
)

trainer = Trainer(
    model=model,        
    args=training_args,      
    train_dataset=train_ds,   
    eval_dataset=val_ds,            
    compute_metrics=compute_metrics,
)

trainer.train()

preds = trainer.predict(test_ds)

y_pred = [np.argmax(array, axis=0) for array in preds.predictions] 
res = classification_report(test_ds.labels, y_pred, digits=3)
print(res)


os.makedirs('./metrics/', exist_ok=True)
save_file = f'./metrics/tuned.txt'
with open(save_file, 'a') as f:
    f.write(f'## {model_name}: {ids_folder}')
    f.write(res)
    f.write('\n\n')


print('done')