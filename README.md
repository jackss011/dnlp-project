# DNLP Project: Text Classification
The folders `specter` and `scidocs` contains the code taken directly from the repository provided by the papers author([Specter](https://github.com/allenai/specter), [SciDocs](https://github.com/allenai/scidocs)) with several modifications. The folder `dblp` folders contains files to create the sparse dataset. The folder `fine-tune` contains files file to test the model fine-tune performance. PCA files are in the `scidocs` folder since they use SciDocs's dataset.

The following instructions can be used to reproduce all results.
> NOTE: Only reproducible on Linux.

## Train specter
### Create Sparse dataset
Download the dataset, extract it and process it to be compatible with specter. Then copy it to specter data folder.
```bash
cd dblp-dataset
wget https://lfs.aminer.cn/lab-datasets/citation/citation-network1.zip
unzip citation-network1.zip -d data2/
python create_dataset.py
cp -r full ../specter/data/training/full
```

### Specter setup
Download required specter files, create Conda environment for specter.
```bash
#  download required files
cd specter
pip install gdown
gdown https://drive.google.com/uc?id=18Ejk3gWTh3aTO2TZxAUc9wBO2xEmE0ab # or follow link and download
tar -xzvf required.tar.gz

conda create --name specter python=3.7 setuptools  
conda activate specter  
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
sudo apt update & sudo apt install build-essential
pip install -r requirements.txt  
python setup.py install
pip install overrides==3.1.0
```

### Train models
Train specter on two datasets:
  - **full**: the dataset create in step 1
  - **sample**: a small sample of 60 papers used to measure baseline performace

The training for the orginal paper model was skipped due to the dataset not being available. The pretrained model from the original authors is used instead in the evaluation phase.

#### Full
Training the full model takes about 30h on a RTX2070Super. (Download already trained below)
```bash
# run in <root>/specter folder
mkdir --verbose --parents data/preproccesed/full

# create training instances
python specter/data_utils/create_training_files.py \
--data-dir data/training/full \
--metadata data/training/full/metadata.json \
--outdir data/preprocessed/full

# run training
./scripts/run-exp-simple.sh -c experiment_configs/simple.jsonnet \
-s model-output-full/ --num-epochs 2 --batch-size 4 \
--train-path data/preprocessed/full/data-train.p --dev-path data/preprocessed/full/data-val.p \
--num-train-instances 55 --cuda-device 0
```

Alternatively it is possible to download already trained models (only sample+full, since original was in the required files).
```bash
pip install gdown
gdown https://drive.google.com/uc?id=1xt0LIp_CLXySntra_a8CUnpdnlQKsfMG # or follow link and download
tar -xzvf models.tar.gz
```

#### Sample
Very quick training due to small dataset
```bash
# run in <root>/specter folder
mkdir --verbose --parents data/preproccesed/sample

# create training instances
python specter/data_utils/create_training_files.py \
--data-dir data/training/sample \
--metadata data/training/sample/metadata.json \
--outdir data/preprocessed/sample

# run training
./scripts/run-exp-simple.sh -c experiment_configs/simple.jsonnet \
-s model-output-sample/ --num-epochs 2 --batch-size 4 \
--train-path data/preprocessed/sample/data-train.p --dev-path data/preprocessed/sample/data-val.p \
--num-train-instances 55 --cuda-device 0
```

### Evaluation (SciDocs)
#### Embeddings
create embeddings of scidocs files for:
  - pretraining original model
  - small sample model
  - full dataset model

```bash
# run in <root>/specter folder
conda activate specter
./scripts/run-embed.sh original model-original  # embed orginal model
./scripts/run-embed.sh sample model-output-sample # embed sample model
./scripts/run-embed.sh full model-output-full # embed 
```
Embeddings are automatically moved to `<root>/scidocs/embdeddings` folder.

Alternatively it is possible to download pre-computed embeddings for all models
```bash
cd scidocs # in scidocs folder
pip install gdown
gdown https://drive.google.com/uc?id=1RgQRfArI382po9aJHxxzYSyQfWSkw9EO # or follow link and download
tar -xzvf embeddings.tar.gz
```

#### Setup SciDocs
Create Conda environment
```bash
cd scidocs

conda create -y --name scidocs python==3.7
conda activate scidocs
conda install -y -q -c conda-forge numpy pandas scikit-learn=0.22.2 jsonlines=3.0.0 tqdm sklearn-contrib-lightning pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
sudo apt update
sudo apt install build-essential
pip install pytrec_eval awscli allennlp==0.9 overrides==3.1.0 seaborn
python setup.py install
```

Download dataset:
```bash
aws s3 sync --no-sign-request s3://ai2-s2-research-public/specter/scidocs/ data/
```

Evaluate embeddings:
```bash
conda activate scidocs
./run-evaluate.sh embeddings/original results/orginal.csv
./run-evaluate.sh embeddings/sample results/sample.csv
./run-evaluate.sh embeddings/full results/full.csv
```
Results will be save in `<root>/scidocs/results`


### [EXTRA]Generate t-SNE visualizations
```bash
# in scidocs folder
python ./scripts/tsne.py embeddings/full/cls.jsonl data/mag/test.csv tsne/full --title Sparse
python ./scripts/tsne.py embeddings/sample/cls.jsonl data/mag/test.csv tsne/sample --title Sample
python ./scripts/tsne.py embeddings/original/cls.jsonl data/mag/test.csv tsne/sample --title Original
```
it is also possible to do a grid search on perplexuty and n_iter by adding `--grid-search` as a parameter of the script.

Results will be in `<root>/scidocs/tsne` folder.

### PCA evaluation
```bash
# in scidocs folder

# create the embdeddings
# requires scidocs dataset
python ./scripts/pca.py data/specter-embeddings
# outputs in pca-embeddings

# run scidocs suite for each value of number of components
./run-evaluate-pca.sh
```
Results are store in `pca-results` folder (values for the report are included in the repo). The `scripts/analyze-pca-results.ipynb` notebook can generate the performance graph included in the paper.

The *number of component* values tested can be modified in the script `scidocs/scripts/pca.py`.

## Fine tune testing
### Create environment
```bash
cd fine-tune
conda create -y --name fine-tune python==3.11
conda activate fine-tune
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install chardet transformers scikit-learn pandas
```

### Download the dataset
```bash
## in fine-tune folder
pip install gdown
gdown https://drive.google.com/uc?id=1_EIfD4nopjPsgRpDH7YxinhDfIgHLZ7y # or follow link and download
tar -xzvf data-fine-tune.tar.gz
```
https://drive.google.com/file/d//view?usp=sharing
### Train and test the models
```bash
# in fine tune folder
# for fixed results
./run-fixed.sh

# for fine tuned results
./run-tuned.sh
```
Results will be stored in the `<root>/fine-tune/metrics` folder.
