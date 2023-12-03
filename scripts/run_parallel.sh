DATASETS=Com2Sense
LOG_FILE="gen_${DATASETS}.log"
GEN=14
# Your API keys (recommend 2 duplicates per key)
API_KEYS=("key1"\ "key1"\ "key2"\ "key2"\ "key3"\ "key3")

python3 parallel_generate.py --api-keys ${API_KEYS} --datasets ${DATASETS} -l ${LOG_FILE} -g ${GEN}
