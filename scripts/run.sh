DATASETS=example
LAYER=0
LOG_FILE="run_${DATASETS}_layer${LAYER}.log"
GEN=2

#python3 parallel_generate.py --api-keys "key1" "key1" "key2" "key2" --datasets $DATASETS -l $LOG_FILE -g $GEN --layer $LAYER & disown
python3 main_inference.py --device-id 0 --dataset-name ${DATASETS} -g ${GEN} --layer ${LAYER}
