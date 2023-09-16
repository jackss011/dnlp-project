from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import argparse


parser = argparse.ArgumentParser(
    prog='Generate t-SNE ',
    description='Generate t-SNE images for the selected ids samples',
    epilog='')

parser.add_argument('embeddings_path')           # positional argument
parser.add_argument('ids_path')      # option that takes a value
parser.add_argument('save_folder', default='./tsne')
parser.add_argument('--grid-search',action='store_true')  
parser.add_argument('-p', '--perplexity', default=15, type=float)
parser.add_argument('-i', '--iters', default=1500, type=int) 
parser.add_argument('-t', '--title', default='t-SNE')   
args = parser.parse_args()

embeddings_path = args.embeddings_path
ids_path = args.ids_path
save_folder = args.save_folder
title = args.title
do_grid = args.grid_search
perplexity_arg = args.perplexity
n_iter_arg = args.iters

os.makedirs(save_folder, exist_ok=True)


print('loading embeddings...')
embeddings = {}

with open(embeddings_path) as f:
    for line in f:
        paper = json.loads(line)
        id = paper['paper_id']
        e = paper['embedding']
        embeddings[id] = np.array(e)


ids = list(embeddings.keys())
print("Num samples:", len(embeddings))


ids_df = pd.read_csv(ids_path)
valid_ids = ids_df['pid'].apply(lambda pid: pid in embeddings.keys())
ids_df_clean = ids_df[valid_ids]

X = np.vstack(ids_df_clean['pid'].apply(lambda pid: embeddings[pid]).to_list() )
y = ids_df_clean['class_label'].to_numpy()
print('embeddings loaded')


tsne_params = {
    "n_components": 2,
    "verbose": 1,
    "random_state": 22,
    "n_jobs": 8,
    "learning_rate": "auto",
    "init": "pca",
}


if not do_grid:
    print(f'running for p:{perplexity_arg} iter:{n_iter_arg}')
    tsne = TSNE(**tsne_params, n_iter=n_iter_arg, perplexity=perplexity_arg)
    # Xf = preprocessing.StandardScaler().fit_transform(X)
    z = tsne.fit_transform(X, y)

    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    num_labels = len(set(y))

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(), s=10,
                    palette=sns.color_palette("Paired", num_labels),
                    data=df).set(title=title)
    plt.savefig(save_folder + f'/p{perplexity_arg}-it{n_iter_arg}.jpeg', format='jpeg', dpi=200)
    print(f"saved to: {save_folder}; p={perplexity_arg}, n_iter={n_iter_arg}")

else:
    param_grid = {
        "perplexity": [1, 2, 5, 10, 15, 20, 25, 30, 40, 50],
        "n_iter": [250, 350, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000],
    }

    print('begin grid:', param_grid)

    for perplexity in param_grid['perplexity']:
        for n_iter in param_grid['n_iter']:
            print(f'running for p:{perplexity_arg} iter:{n_iter_arg}')
            tsne1 = TSNE(**tsne_params, perplexity=perplexity, n_iter=n_iter)
            z = tsne1.fit_transform(X, y)

            df = pd.DataFrame()
            df["y"] = y
            df["comp-1"] = z[:,0]
            df["comp-2"] = z[:,1]
            num_labels = len(set(y))

            plt.clf()
            sns.scatterplot(x="comp-1", y="comp-2", 
                            hue=df.y.tolist(), s=10, palette=sns.color_palette("Paired", num_labels),
                            data=df).set(title=title)
            
            plt.savefig(save_folder + f'/p{perplexity}-it{n_iter}.jpeg', format='jpeg', dpi=200)
            print(f"saved to: {save_folder}; p={perplexity}, n_iter={n_iter}")