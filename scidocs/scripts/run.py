
from scidocs.paths import DataPaths
from scidocs import get_scidocs_metrics

import argparse
import json
import pandas as pd
import os
from datetime import datetime


pd.set_option('display.max_columns', None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '--classification-embeddings-path', dest='cls', help='path to classification related embeddings (mesh and mag)')
    parser.add_argument('--user-citation', '--user_activity_and_citations_embeddings_path', dest='user_citation', help='path to user activity embeddings (coview, copdf, cocite, citation)')
    parser.add_argument('--recomm', '--recomm_embeddings_path', dest='recomm', help='path to recommender related embeddings')
    parser.add_argument('--val_or_test', default='test', help='whether to evaluate scidocs on test data (what is reported in the specter paper) or validation data (to tune hyperparameters)')
    parser.add_argument('--n-jobs', default=12, help='number of parallel jobs for classification (related to mesh/mag metrics)', type=int)
    parser.add_argument('--cuda-device', default=-1, help='specify if you want to use gpu for training the recommendation model; -1 means use cpu')
    parser.add_argument('--data-path', default=None, help='path to the data directory where scidocs files reside. If None, it will default to the `data/` directory')
    parser.add_argument('--output-file', default=None, help='path where to save results')
    args = parser.parse_args()

    data_paths = DataPaths(args.data_path)

    if args.output_file:
        output_folder = os.path.dirname(args.output_file)
        os.makedirs(output_folder, exist_ok=True)

        # tag = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = os.path.basename(args.output_file)
        if not filename.endswith('.csv'):
            filename += '.csv'
        # tagged_filename = f"{tag}__{filename}"
        tagged_filename = f"{filename}"
        tagged_filepath = os.path.join(output_folder, tagged_filename)
        print("Will save to:", tagged_filepath)

    scidocs_metrics = get_scidocs_metrics(
        data_paths,
        args.cls,
        args.user_citation,
        args.recomm,
        val_or_test=args.val_or_test,
        n_jobs=args.n_jobs,
        cuda_device=args.cuda_device
    )

    flat_metrics = {}
    for k, v in scidocs_metrics.items():
        if not isinstance(v, dict):
            flat_metrics[k] = v
        else:
            for kk, vv in v.items():
                key = k + '-' + kk
                flat_metrics[key] = vv
    df = pd.DataFrame(list(flat_metrics.items())).T

    print("Results:")
    print(df)

    if args.output_file:
        df.to_csv(tagged_filepath)
        print("Saved to:", tagged_filepath)
    

if __name__ == '__main__':
    main()




