DATASET=example
LAYER=0
GEN=2
LOG_FILE="infer_${DATASET}_layer${LAYER}.log"
DEVICE_ID=0

python main_inference.py --device-id ${DEVICE_ID} --dataset-name ${DATASET} --layer ${LAYER} --gen ${GEN} --log ${LOG_FILE}