# Install environment
git clone https://github.com/allenai/specter.git
cd specter
wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz
mkdir model-original
tar -xzvf archive.tar.gz -C ./model-original

conda create --name specter python=3.7 setuptools  
conda activate specter  
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
sudo apt update
sudo apt install build-essential
pip install -r requirements.txt  
python setup.py install
pip install overrides==3.1.0


# INCLUDED IN SCIDOCS
## run benchmark
cd scidocs
conda activate scidocs

python scripts/run.py --cls data/specter-embeddings/cls.jsonl --user-citation data/specter-embeddings/user-citation.jsonl --recomm data/specter-embeddings/recomm.jsonl --val_or_test test --n-jobs 12 --cuda-device 0

# PRETRAINED MODEL
## Embedd
cd specter
mkdir ../scidocs/data/embedded && mkdir ../scidocs/data/embedded/original
conda activate specter

python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_mag_mesh.json \
--model ./model-original/model.tar.gz \
--output-file ../scidocs/data/embedded/original/cls.jsonl \
--vocab-dir ./model-original/data/vocab/ \
--batch-size 16 \
--cuda-device 0

python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_recomm.json \
--model ./model-original/model.tar.gz \
--output-file ../scidocs/data/embedded/original/recomm.jsonl \
--vocab-dir ./model-original/data/vocab/ \
--batch-size 16 \
--cuda-device 0

python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_view_cite_read.json \
--model ./model-original/model.tar.gz \
--output-file ../scidocs/data/embedded/original/user-citation.jsonl \
--vocab-dir ./model-original/data/vocab/ \
--batch-size 8 \
--cuda-device 0


## metrics
cd scidocs
conda activate scidocs

python scripts/run.py --cls data/embedded/original/cls.jsonl \
--user-citation data/embedded/original/user-citation.jsonl \
--recomm data/embedded/original/recomm.jsonl \
--val_or_test test --n-jobs 12 --cuda-device 0

# SIMPLE TRAINED MODEL
cd specter
conda activate specter
mkdir data/preproccesed && mkdir data/preproccesed/sample

## Train
python specter/data_utils/create_training_files.py \
--data-dir data/training/sample \
--metadata data/training/sample/metadata.json \
--outdir data/preprocessed/sample

./scripts/run-exp-simple.sh -c experiment_configs/simple.jsonnet \
-s model-output-sample/ --num-epochs 2 --batch-size 4 \
--train-path data/preprocessed/sample/data-train.p --dev-path data/preprocessed/sample/data-val.p \
--num-train-instances 55 --cuda-device 0

# Embedd
python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_mag_mesh.json \
--model ./model-output-sample/model.tar.gz \
--output-file ../scidocs/data/embedded/sample/cls.jsonl \
--vocab-dir ./model-output-sample/vocabulary/ \
--batch-size 8 \
--cuda-device 0

python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_recomm.json \
--model ./model-output-sample/model.tar.gz \
--output-file ../scidocs/data/embedded/sample/recomm.jsonl \
--vocab-dir ./model-output-sample/vocabulary/ \
--batch-size 8 \
--cuda-device 0

python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_view_cite_read.json \
--model ./model-output-sample/model.tar.gz \
--output-file ../scidocs/data/embedded/sample/user-citation.jsonl \
--vocab-dir ./model-output-sample/vocabulary/ \
--batch-size 8 \
--cuda-device 0

## Metrics
cd scidocs
conda activate scidocs

python scripts/run.py --cls data/embedded/sample/cls.jsonl \
--user-citation data/embedded/sample/user-citation.jsonl \
--recomm data/embedded/sample/recomm.jsonl \
--val_or_test test --n-jobs 12 --cuda-device 0

python scripts/run.py --cls data/embedded/full/cls.jsonl \
--user-citation data/embedded/full/user-citation.jsonl \
--recomm data/embedded/full/recomm.jsonl \
--val_or_test test --n-jobs 12 --cuda-device 0


## Train
## Eval
### Embed
### Compute