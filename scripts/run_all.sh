DATASET=example
GEN=2
DEVICE_ID=0
START_LAYER=0
MAX_DEPTH=2

# Your API keys (recommend 2 duplicates per key)
API_KEYS=("key1"\ "key1"\ "key2"\ "key2"\ "key3"\ "key3")

for ((i=START_LAYER; i<=MAX_DEPTH+1; i++)); do
    LAYER=$i
    GEN_LOG_FILE="gen_${DATASET}_layer${LAYER}.log"
    INFER_LOG_FILE="infer_${DATASET}_layer${LAYER}.log"
    python parallel_generate.py --api-keys ${API_KEYS} --datasets ${DATASET} -l ${GEN_LOG_FILE} -g ${GEN} --layer ${LAYER}
    python main_inference.py --device-id ${DEVICE_ID} --dataset-name ${DATASET} --layer ${LAYER} --gen ${GEN} --log ${INFER_LOG_FILE}
done
