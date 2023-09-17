import pandas as pd
import json
import sys
import os

from transformers import AutoTokenizer, AutoModel
import torch
import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, homogeneity_completeness_v_measure, pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset(ids_path, metadata):
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


if '--mag' in sys.argv:
    ids_folder = 'mag'
elif '--mesh' in sys.argv:
    ids_folder = 'mesh'
else:
    raise ValueError("select either --mag or --mesh datasets")
print("selected:", ids_folder)

with open('./data/cls-metadata.json', encoding='utf-8') as f:
  metadata = json.load(f)

train_X, train_y = load_dataset(f'./data/{ids_folder}/train.csv', metadata)
val_X, val_y = load_dataset(f'./data/{ids_folder}/val.csv', metadata)
test_X, test_y = load_dataset(f'./data/{ids_folder}/test.csv', metadata)

num_labels = len(set(train_y))
print(num_labels, len(train_X), len(val_X), len(test_X))


model_name = 'allenai/specter'
if '--scibert' in sys.argv:
    model_name = 'allenai/scibert_scivocab_uncased'
print('using model', model_name)
model_short_name = model_name.split('/')[1]


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


embedded_folder = f'./preprocessed/pca/{model_short_name}_{ids_folder}'

if '--load' not in sys.argv:
    os.makedirs(embedded_folder, exist_ok=True)

    train_encoded = encode(model, tokenizer, train_X)
    test_encoded = encode(model, tokenizer, test_X)

    torch.save(train_encoded, embedded_folder + '/train.p')
    torch.save(test_encoded, embedded_folder + '/test.p')
else:
    train_encoded = torch.load(embedded_folder + '/train.p')
    test_encoded = torch.load(embedded_folder + '/test.p')

print(train_encoded.size())


n_components = list([5, 10, 25, 50, 75] + list(range(100, 701, 50)))
variance_explained = []

for nc in n_components:
    pca = PCA(n_components=nc)
    # std = StandardScaler()
    # X = std.fit_transform(train_encoded)
    X = train_encoded
    pca.fit_transform(X)
    ve = sum(pca.explained_variance_ratio_)
    variance_explained.append(ve)
    print(nc, ve)

plt.tight_layout()
sns.lineplot(x=n_components, y=variance_explained, palette=['o'])
plt.ylabel("% variance explained")
plt.xlabel('number of components')
plt.title('MAG PCA')
plt.grid()
plt.savefig("out.jpg", format='jpeg', dpi=300)

quit()

pca = PCA(n_components=200)
X_train = train_encoded
# X_train = StandardScaler().fit_transform(X_train)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(test_encoded)

# m = pairwise_distances(X_train, X_train).reshape(-1)
# print(m.mean(), m.std())

# X_train = train_encoded

grid_params = {
    "eps": [6, 8, 10, 12, 14, 16],
    "min_samples": [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 40, 50, 100, 150]
}

# for eps in grid_params['eps']:
#     for min_samples in grid_params['min_samples']:
#         clustering = DBSCAN(eps=eps, min_samples=min_samples)
#         clustering.fit(X_train)
#         print(eps, min_samples)
#         print(len(set(clustering.labels_)))
#         print(list(clustering.labels_).count(-1)/len(clustering.labels_))
#         # print(silhouette_score(X_train, clustering.labels_))
#         print(homogeneity_completeness_v_measure(clustering.labels_, train_y), '\n')


# clusterer = KMeans(n_clusters=100, n_init=20)
# clusterer.fit(X_train)
# print(homogeneity_completeness_v_measure(clusterer.labels_, train_y))
# print(silhouette_score(X_train, labels=clusterer.labels_))


