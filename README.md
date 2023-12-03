# Multiple Choice Question Answering with Maieutic Prompting

This is the code repository for our project **Multiple Choice Question Answering with Maieutic Prompting**, which is based on the paper **Maieutic Prompting: Logically Consistent Reasoning with Recursive Explanations**, whose implementation is available in [this GitHub repo](https://github.com/jaehunjung1/Maieutic-Prompting).

## Requirements

```shell
pip install -r requirements.txt
```

## Generation

In order to generate maieutic tree for a given set of questions, run `main_generate.py`. We provide the code and arguments to run generation for the 3 datasets - Com2Sense, CSQA 2.0, CREAK used in the paper. To simply run generation for the dev split of these datasets, run

```shell
python main_generate.py --datasets=${DATASET_NAME} --api-key=${API_KEY} -g ${GEN} --layer ${LAYER}
```

The code reads dataset file in `./data/{datset_name}/{gen}_gen/dev.Q.json` as input and outputs the pickled list of lists of trees in `./data/{dataset_name}/{gen}_gen/dev_layer{layer}.G.pkl` for the specified layer.

- We use `treelib` library to represent maieutic tree. For further documentation please refer to [Official Doc](https://treelib.readthedocs.io/en/latest/).

The generation code requires PaLM API key as a command-line argument. For easier access to our method, we provide pre-generated files of maieutic trees for each dataset in `./data/{dataset_name}/1_gen/dev_layer{layer}.G.pkl`.

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
python main_inference.py --device-id=${DEVICE_ID} --dataset-name=${DATASET_NAME} -g ${GEN} --layer ${LAYER}
```

`device_id` denotes the id of GPU to load the verifier model.
If `layer` is 0, the code reads dataset file in `./data/{dataset_name}/{gen}_gen/dev.Q.json` and the maieutic tree file in `./data/{dataset_name}/{gen}_gen/dev_layer0.G.pkl` and generates the result file in `./data/{dataset_name}/{gen}_gen/dev_layer0.C.pkl`.
Otherwise, it reads the dataset file, the maieutic tree file in `./data/{dataset_name}/{gen}_gen/dev_layer{layer}.G.pkl`, _and the result file of the previous layer_ in `./data/{dataset_name}/{gen}_gen/dev_layer{layer-1}.C.pkl`.

For convenience, you can also run

```shell
./scripts/run_inference.sh
```

Make sure that the arguments are properly set in `./scripts/run_inference.sh`.

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

## Run all

Since maieutic prompting for MCQA requires sequential execution of multiple generation and inference steps, for convenience, we provide a shell script to run all the steps sequentially. Use the following commands to run all the generation and inference steps. Make sure to set the variables in `./scripts/run_all.sh` to the correct values before running.

### Run in the foreground

```shell
bash ./scripts/run_all.sh
```

### Run in the background

```shell
bash ./scripts/run_all.sh & disown
```

### Cancel the run in the background

```shell
kill -9 <pid>
```

where `pid` is the PID of the bash process, which can be obtained by

```shell
ps aux | grep run_all.sh
```

This will kill all the used processes, so you don't have to do anything else.

### Clean up cache results

```shell
bash ./scripts/clean.sh -d <dataset> -g <gen>
```

where `dataset` is the dataset and `gen` is the generation you want to clear.
**Note that the result files are stored in "`./data/<dataset>/<gen>_gen/`".**
