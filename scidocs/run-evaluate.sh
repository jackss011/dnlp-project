python scripts/run.py --cls $1/cls.jsonl \
--user-citation $1/user-citation.jsonl \
--recomm $1/recomm.jsonl \
--output-file $2 \
--val_or_test test --n-jobs 12 --cuda-device 0