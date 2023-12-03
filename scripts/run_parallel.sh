DATASETS=example
LAYER=0
LOG_FILE="gen_${DATASETS}_layer${LAYER}.log"
GEN=2
# Your API keys (recommend 2 duplicates per key)
API_KEYS=("key1"\ "key1"\ "key2"\ "key2"\ "key3"\ "key3")

python3 parallel_generate.py --api-keys ${API_KEYS} --datasets ${DATASETS} -l ${LOG_FILE} -g ${GEN} --layer ${LAYER} & disown
