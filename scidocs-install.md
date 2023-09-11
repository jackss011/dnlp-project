# Install environment
git clone https://github.com/allenai/scidocs.git
cd scidocs

conda create -y --name scidocs python==3.7
conda activate scidocs
conda install -y -q -c conda-forge numpy pandas scikit-learn=0.22.2 jsonlines=3.0.0 tqdm sklearn-contrib-lightning pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
sudo apt update
sudo apt install build-essential
pip install pytrec_eval awscli allennlp==0.9 overrides==3.1.0
python setup.py install

# Download dataset (4.1GB)
aws s3 sync --no-sign-request s3://ai2-s2-research-public/specter/scidocs/ data/

# Run test
python scripts/run.py --cls data/specter-embeddings/cls.jsonl --user-citation data/specter-embeddings/user-citation.jsonl --recomm data/specter-embeddings/recomm.jsonl --val_or_test test --n-jobs 12 --cuda-device 0


# RESUSLTS: ORIGINAL SPECTER embeddings
       0        1            2             3            4             5   \
0  mag-f1  mesh-f1  co-view-map  co-view-ndcg  co-read-map  co-read-ndcg
1   81.95    86.44        83.63          91.5        84.46         92.39

         6          7            8             9                10  \
0  cite-map  cite-ndcg  co-cite-map  co-cite-ndcg  recomm-adj-NDCG
1      88.3      94.88        88.11         94.77            53.83

               11
0  recomm-adj-P@1
1           19.77