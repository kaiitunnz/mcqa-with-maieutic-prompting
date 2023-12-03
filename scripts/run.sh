GEN=2
DATASETS=Com2Sense
LOG_FILE="gen_${DATASETS}.log"
API_KEY="api_key"

python main_generate.py --api-key ${API_KEY} --datasets $DATASETS -l $LOG_FILE -g $GEN