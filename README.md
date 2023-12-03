# Maieutic Prompting with an open-source LLM

This is an adaptation of the implementation code of the paper **Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations** available in this [GitHub repo](https://github.com/jaehunjung1/Maieutic-Prompting) to use an open-source LLM instead of OpenAI's deprecated text-davinci-001.

## Requirements

```shell
pip install -r requirements.txt
```

## Start the LLM server

To start an LLM server, run

```shell
python model_server/server.py --model-name ${MODEL_NAME} --config-file ${CONFIG}
```

It will start a model server that serves the Hugging Face model specified by `model_name` with the server configuration specified by `config`. You can see an example of the configuration file in `model_server/config.json`.

We provide a shell script to start two LLM servers. To do so, run

```shell
./scripts/start_servers.sh
```

However, the server will terminate once the shell exits. To allow closing the shell without terminating the servers, run

```shell
bash ./scripts/start_servers.sh & disown
```

## Generation

In order to generate maieutic tree for a given set of questions, run `main_generate.py`. We provide the code and arguments to run generation for the 3 datasets - Com2Sense, CSQA 2.0, CREAK used in the paper. To simply run generation for the dev split of these datasets, run

```shell
python main_generate.py --dataset_name=${DATASET_NAME} --server-config=${CONFIG} -g ${GEN}
```

The code reads dataset file in `./data/{datset_name}/{gen}_gen/dev.Q.json` as input and outputs the pickled list of trees in `./data/{dataset_name}/{gen}_gen/dev.G.pkl`. It will make requests to the server according to the configuration file specified by `config`.

- We use `treelib` library to represent maieutic tree. For further documentation please refer to [Official Doc](https://treelib.readthedocs.io/en/latest/).

For easier access to our method, we provide pre-generated files of maieutic tree for each dataset in `./data/{dataset_name}/1_gen/dev.G.pkl`.

Alternatively, for convenience, we provide a script that executes `main_generate.py` for you. See `./scripts/run.sh` for details and change the arguments as you want. To run in the foreground, run

```shell
./scripts/run.sh
```

Otherwise, to run in the background, run

```shell
bash ./scripts/run.sh & disown
```

This will allow you to close the terminal while generating the trees.

## Inference

To run inference, run

```shell
python main_inference.py --device_id=${DEVICE_ID} --dataset-name=${DATASET_NAME} -g ${GEN}
```

`device_id` denotes the id of GPU to load the verifier model. The code reads dataset file in `./data/{datset_name}/{gen}_gen/dev.Q.json` and maieutic tree file in `./data/{dataset_name}/{gen}_gen/dev.G.pkl`

For convenience, you can also run

```shell
./scripts/run_inference.sh
```

Make sure that the arguments are properly set in `./scripts/run_inference.sh`.

## Parallel generation

You can generate maieutic trees for multiple datasets in parallel. To do so, set the arguments accordingly and run

```shell
./scripts/run_parallel.sh
```

However, the generation processes will terminate once the shell exits. To allow closing the shell without terminating the processes, run

```shell
bash ./scripts/run_parallel.sh & disown
```

## Clean up cache results

To clean up cache results for `dataset` and `gen`, run

```shell
./scripts/clean.sh -d ${DATASET} -g${GEN}
```

This will delete all the pickle and log files found in `./data/{dataset}/{gen}_gen`.
