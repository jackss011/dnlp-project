import os
import sys
import json
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


embeddings_folder = 'embeddings/original'
output_folder = 'pca-embeddings'
n_jobs = 6


def read_train_ids(files):
    print('reading ids:', files)
    train_ids = set()

    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(file)
            train_ids.update(set(df['pid'].to_list()))

        elif file.endswith('.qrel'):
            with open(file, encoding='utf-8') as f:
                lines = f.readlines()
            parts = [line.split(' ') for line in lines]
            ids1 = set([p[0] for p in parts])
            ids2 = set([p[2] for p in parts])
            train_ids.update(ids1)
            train_ids.update(ids2)

    print(f'read {len(train_ids)} train ids')
    return train_ids


def read_embeddings(papers_file, train_ids):
    embeddings = {}
    titles = {}

    print(f'loading {papers_file} embeddings...')
    with open(papers_file, encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            pid = paper['paper_id']
            e = paper['embedding']
            embeddings[pid] = np.array(e).reshape(1, -1)
            titles[pid] = paper['title']
    print(f'loaded {papers_file} embeddings')

    train_embeddings = [embeddings[id] for id in train_ids if embeddings.get(id) is not None]
    X_train = np.vstack(train_embeddings)

    print("Num papers for PCA training:", len(train_embeddings))
    print("Train matrix shape", X_train.shape)
    return embeddings, titles, X_train


def save_pca(embeddings, titles, X_train, n_components, save_file):
    print(f'training PCA[{n_components}] ...')
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    print(f'trained PCA[{n_components}]')

    save_folder = os.path.dirname(save_file)
    os.makedirs(save_folder, exist_ok=True)

    print(f'inferring PCA[{n_components}] ...')
    with open(save_file, 'w', encoding='utf-8') as f:
        for pid in list(embeddings.keys()):
            x = embeddings[pid]
            y = pca.transform(x)
            line = json.dumps({"paper_id": pid, "title": titles[pid], "embedding": list(y.reshape(-1))})
            f.write(line + '\n')
    print(f'inferred PCA[{n_components}]')

# n_components_list = [100]
n_components_list = [50, 75, 100, 150, 200, 300, 400, 500, 600, 700]


def run_task(papers_file_name, train_ids_files):
    print(f"\n----- {papers_file_name} -----")

    train_ids = read_train_ids(train_ids_files)
    embeddings, titles, X_train = read_embeddings(f"{embeddings_folder}/{papers_file_name}", train_ids)

    def save_pca_task(nc):
        save_file = f"{output_folder}/nc{nc}/{papers_file_name}"
        save_pca(embeddings, titles, X_train, nc, save_file)

    if n_jobs == 1:
        print('running in single thread mode (may be slow)')
        for n_components in n_components_list:
            save_pca_task(n_components)
    elif n_jobs > 1:
        print(f'running in multithread mode (workers: {n_jobs})')
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            executor.map(save_pca_task, n_components_list)
    else:
        raise ValueError(f"invalid number of jobs: {n_jobs}")

    print('done\n')


run_task('user-citation.jsonl', ['data/cocite/val.qrel', 'data/cite/val.qrel', 'data/cocite/val.qrel', 'data/coread/val.qrel', 'data/coview/val.qrel'])
run_task('cls.jsonl', ['data/mag/train.csv', 'data/mesh/train.csv'])
run_task('recomm.jsonl', ['data/recomm/train.csv'])



# train_ids_files = ['data/mag/train.csv', 'data/mesh/train.csv']
# n_components = 100
# embeddings = {}
# titles = {}

# print('loading cls embeddings...')
# with open(embeddings_folder + '/cls.jsonl', encoding="utf-8") as f:
#     for line in f:
#         paper = json.loads(line)
#         pid = paper['paper_id']
#         e = paper['embedding']
#         embeddings[pid] = np.array(e).reshape(1, -1)
#         titles[pid] = paper['title']
# print('loaded cls embeddings')


# train_ids = set()

# for train_ids_file in train_ids_files:
#     df = pd.read_csv(train_ids_file)
#     train_ids.update(set(df['pid'].to_list()))


# train_embeddings = [embeddings[id] for id in train_ids if embeddings.get(id) is not None]
# X_train = np.vstack(train_embeddings)

# print("Num papers for PCA training:", len(train_embeddings))
# print("Train matrix shape", X_train.shape)

# print(f'training PCA (n={n_components})...')
# pca = PCA(n_components=n_components)
# pca.fit(X_train)
# print('PCA trained')

# print('inferring PCA...')
# results_folder = output_folder + f'/nc{n_components}'
# os.makedirs(results_folder, exist_ok=True)

# with open(results_folder + '/cls.json', 'w', encoding='utf-8') as f:
#     for pid in embeddings.keys():
#         x = embeddings[pid]
#         y = pca.transform(x)
#         line = json.dumps({"paper_id": pid, "title": titles[pid], "embedding": list(y.reshape(-1))})
#         f.write(line + '\n')
# print('inferred PCA')



