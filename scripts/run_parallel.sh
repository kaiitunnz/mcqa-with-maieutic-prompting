GEN=2

python main_generate.py --datasets Com2Sense -l gen_Com2Sense.log -g $GEN --server-config model_server/config1.json &
python main_generate.py --datasets CREAK -l gen_CREAK.log -g $GEN --server-config model_server/config1.json &
python main_generate.py --datasets CSQA2 -l gen_CSQA2.log -g $GEN --server-config model_server/config2.json &
