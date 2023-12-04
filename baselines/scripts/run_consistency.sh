DATASETS="CommonsenseQA ARCeasy ARCchallenge"
API_KEYS=("key1"\ "key1"\ "key2"\ "key2"\ "key3"\ "key3")

for dataset in $DATASETS; do
    python self_consistency.py --api-keys ${API_KEYS} --dataset-name $dataset --log "log/${dataset}_selfcon.log"
done
