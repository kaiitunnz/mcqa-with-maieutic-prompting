DATASETS=Com2Sense
LOG_FILE="gen_$DATASETS.log"
GEN=2

python main_generate.py --datasets $DATASETS -l $LOG_FILE -g $GEN & disown
