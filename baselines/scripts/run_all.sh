DATASETS="CommonsenseQA ARCeasy ARCchallenge"

for dataset in $DATASETS; do
    python direct.py -d $dataset -l log/${dataset}_direct.log
    python cot.py -d $dataset -l log/${dataset}_cot.log
done
