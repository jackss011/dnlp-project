mkdir --parents --verbose ../scidocs/embeddings/$1


python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_mag_mesh.json \
--model ./$2/model.tar.gz \
--output-file ../scidocs/embeddings/$1/cls.jsonl \
--vocab-dir ./$2/vocabulary/ \
--batch-size 8 \
--cuda-device 0

python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_recomm.json \
--model ./$2/model.tar.gz \
--output-file ../scidocs/embeddings/$1/recomm.jsonl \
--vocab-dir ./$2/vocabulary/ \
--batch-size 8 \
--cuda-device 0

python ./scripts/embed.py \
--metadata ../scidocs/data/paper_metadata_view_cite_read.json \
--model ./$2/model.tar.gz \
--output-file ../scidocs/embeddings/$1/user-citation.jsonl \
--vocab-dir ./$2/vocabulary/ \
--batch-size 8 \
--cuda-device 0