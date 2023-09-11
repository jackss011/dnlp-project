import pandas as pd
import json
import sys

train_df = pd.read_csv('./data/train.csv')
val_df = pd.read_csv('./data/val.csv')
test_df = pd.read_csv('./data/test.csv')

with open('./data/metadata.json', encoding='utf-8') as f:
  metadata = json.load(f)

def get_title_and_abs(pid):
    paper = metadata[pid]
    title, abstract = paper['title'], paper['abstract']
    return title + ' [SEP] ' + (abstract or '')

train_X = train_df['pid'].apply(get_title_and_abs).tolist()
train_y = train_df['class_label'].tolist()

val_X = val_df['pid'].apply(get_title_and_abs).tolist()
val_y = val_df['class_label'].tolist()

test_X = test_df['pid'].apply(get_title_and_abs).tolist()
test_y = test_df['class_label'].tolist()

num_labels = len(set(train_y))
print(num_labels, len(train_X), len(val_X), len(test_X))

from transformers import AutoTokenizer, AutoModel
import torch
import tqdm

model_name = 'allenai/specter'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def encode(model, tokenizer, X, batch_size=8, device='cuda'):
    model.eval()
    model.to(device)
    all_embeddings = []

    for start_index in tqdm.tqdm(range(0, len(X), batch_size), desc='batches'):
        batch = X[start_index:start_index+batch_size]
        features = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        features = {k: v.to(device) for k, v in features.items()}

        with torch.no_grad():
            output = model(**features)
            embeddings = output.last_hidden_state[:, 0, :].detach()
            all_embeddings.extend(embeddings.cpu())

        # if start_index > 20: break

    all_embeddings = torch.stack(all_embeddings)
    return all_embeddings

if '--load' not in sys.argv:
    train_encoded = encode(model, tokenizer, train_X)
    test_encoded = encode(model, tokenizer, test_X)
    torch.save(train_encoded, './tmp/train.p')
    torch.save(test_encoded, './tmp/test.p')
else:
    train_encoded = torch.load('./tmp/train.p')
    test_encoded = torch.load('./tmp/test.p')

print(train_encoded.size())



from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

# train_y = train_y[:32]
# test_y = test_y[:32]

classifier = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 100, ))
classifier.fit(train_encoded, train_y)
pred_y = classifier.predict(test_encoded)

print(classification_report(test_y, pred_y))