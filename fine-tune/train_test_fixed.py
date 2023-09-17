import pandas as pd
import json
import sys
import os

from transformers import AutoTokenizer, AutoModel
import torch
import tqdm

def load_dataset(ids_path, metadata):
    """
        ids_path: path to scidocs ids file, used for separating train test split
        metadata: dict containing papers metadata
        @returns (X, y): X is list of strings containing paper abstract concatenated
    """
    df = pd.read_csv(ids_path)

    invalid_ids = df['pid'].apply(lambda pid: pid not in metadata.keys())
    df = df[~invalid_ids]

    def get_title_and_abs(pid):
        paper = metadata.get(pid)
        title, abstract = paper['title'], paper['abstract']
        return title + ' [SEP] ' + (abstract or '')
    
    X = df['pid'].apply(get_title_and_abs).tolist()
    y = df['class_label'].tolist()
    return X, y

# select which task
if '--mag' in sys.argv:
    ids_folder = 'mag'
elif '--mesh' in sys.argv:
    ids_folder = 'mesh'
else:
    raise ValueError("select either --mag or --mesh datasets")
print("selected:", ids_folder)

# selected the model scibert/specter
model_name = 'allenai/specter'
if '--scibert' in sys.argv:
    model_name = 'allenai/scibert_scivocab_uncased'
print('using model', model_name)

# read metadata
with open('./data/cls-metadata.json', encoding='utf-8') as f:
  metadata = json.load(f)

# load all splits
train_X, train_y = load_dataset(f'./data/{ids_folder}/train.csv', metadata)
val_X, val_y = load_dataset(f'./data/{ids_folder}/val.csv', metadata)
test_X, test_y = load_dataset(f'./data/{ids_folder}/test.csv', metadata)

num_labels = len(set(train_y))
# print(num_labels, len(train_X), len(val_X), len(test_X))


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# encode tha paper title abstract using the selected model
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


# load from file if already computed
if '--load' not in sys.argv:
    train_encoded = encode(model, tokenizer, train_X)
    test_encoded = encode(model, tokenizer, test_X)
    os.makedirs(f'./preprocessed/{ids_folder}', exist_ok=True)
    torch.save(train_encoded, f'./preprocessed/{ids_folder}/train.p')
    torch.save(test_encoded, f'./preprocessed/{ids_folder}/test.p')
else:
    train_encoded = torch.load(f'./preprocessed/{ids_folder}/train.p')
    test_encoded = torch.load(f'./preprocessed/{ids_folder}/test.p')

print(train_encoded.size())


# run the classifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(max_iter=2000, hidden_layer_sizes=(1000, ))
classifier.fit(train_encoded, train_y)
pred_y = classifier.predict(test_encoded)

res = classification_report(test_y, pred_y, digits=3)
print(res)

# save results to file
os.makedirs('./metrics/', exist_ok=True)
save_file = f'./metrics/fixed.txt'
with open(save_file, 'a') as f:
    f.write(f'## {model_name}: {ids_folder}\n')
    f.write(res)
    f.write('\n\n')