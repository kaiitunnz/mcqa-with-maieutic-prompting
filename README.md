# Maieutic Prompting with PaLM 2

This is an adaptation of the implementation code of the paper **Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations** available in this [GitHub repo](https://github.com/jaehunjung1/Maieutic-Prompting) to use PaLM 2 via Google's PaLM API instead of OpenAI's deprecated text-davinci-001.

## Requirements

```shell
pip install -r requirements.txt
```

## Generation

In order to generate maieutic tree for a given set of questions, run `main_generate.py`. We provide the code and arguments to run generation for the 3 datasets - Com2Sense, CSQA 2.0, CREAK used in the paper. To simply run generation for the dev split of these datasets, run

```shell
python main_generate.py --datasets=${DATASET_NAME} --api-key=${API_KEY} -g ${GEN}
```

The code reads dataset file in `./data/{datset_name}/{gen}_gen/dev.Q.json` as input and outputs the pickled list of trees in `./data/{dataset_name}/{gen}_gen/dev.G.pkl`.

- We use `treelib` library to represent maieutic tree. For further documentation please refer to [Official Doc](https://treelib.readthedocs.io/en/latest/).

The generation code requires PaLM API key as a command-line argument. For easier access to our method, we provide pre-generated files of maieutic tree for each dataset in `./data/{dataset_name}/1_gen/dev.G.pkl`.

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
python main_inference.py --device-id=${DEVICE_ID} --dataset-name=${DATASET_NAME} -g ${GEN}
```

`device_id` denotes the id of GPU to load the verifier model. The code reads dataset file in `./data/{dataset_name}/{gen}_gen/dev.Q.json` and maieutic tree file in `./data/{dataset_name}/{gen}_gen/dev.G.pkl`

## Parallel generation

We provide the code for generating maieutic trees using multiple API keys in parallel. We recommend you to see `./scripts/run_parallel.sh` for details and change the arguments as you want. To run in the foreground, run

```shell
./scripts/run_parallel.sh
```

Otherwise, to run in the background, run

```shell
bash ./scripts/run_parallel.sh & disown
```

Or you can directly execute `parallel_generate.py`, but make sure to properly set the command-line arguments.
