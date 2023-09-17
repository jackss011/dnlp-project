for d in pca-embeddings/*/ ; do
    nc=$(basename "$d")
    #./run-evaluate.sh pca-embeddings/nc100 pca-results/nc100.csv
    ./run-evaluate.sh $d "pca-results/$nc.csv"
done