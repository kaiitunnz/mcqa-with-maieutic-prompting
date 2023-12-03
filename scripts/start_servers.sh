CONFIG_DIR=model_server

python model_server/server.py --config-file "${CONFIG_DIR}/config1.json" &
python model_server/server.py --config-file "${CONFIG_DIR}/config2.json" &