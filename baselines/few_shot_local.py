import os
import re
import sys
import time
from tqdm import tqdm

import jsonlines
from transformers import GenerationConfig

sys.path.append("..")  # Put path to model_server here.

from model_server.client import ModelClient

DATASET = "Com2Sense"

generation_config = GenerationConfig(
    max_new_tokens=64,
    do_sample=True,
    temperature=0.5,
    top_p=1,
)


def main():
    log_file = open(f"time_nocot_{DATASET}_local.log", "w")
    err_file = open(f"time_error_nocot_{DATASET}_local.log", "w")

    model_client = ModelClient()
    with open(f"data/{DATASET}/few_shot.txt", "r") as f:
        prompt_prefix = f.read()

    with jsonlines.open(f"data/{DATASET}/dev.Q.json", "r") as reader:
        data = list(reader)
    correct = 0
    start_time = time.time()
    for i, sample in tqdm(enumerate(data), file=(None if log_file is None else open(os.devnull, "w"))):
        sent = sample["Q"]
        label = sample["A"]
        prompt = f"{prompt_prefix}Q: {sent}\nA: "
        while True:
            result = model_client.generate(prompt, generation_config, 1)[0].text

            matched = re.search("(true|false)", result.lower())
            if matched:
                pred = True if matched.groups()[0] == "true" else False
                if pred == label:
                    correct += 1
                print(
                    {
                        "i": i,
                        "result": pred == label,
                        "label": label,
                        "pred": pred,
                        "Q": sent,
                        "A": result,
                    },
                    file=log_file,
                    flush=True,
                )
                break
            print("result:", result, file=err_file, flush=True)
            print("Retrying...", file=err_file, flush=True)
    elapsed_time = time.time() - start_time

    print(f"Correct: {correct}/{len(data)}", file=log_file, flush=True)
    print(f"Accuracy: {correct/len(data)}", file=log_file, flush=True)
    print(f"Finished in: {elapsed_time} seconds", file=log_file, flush=True)
    print(f"Samples per minute: {len(data) / (elapsed_time) * 60}", file=log_file, flush=True)

    log_file.close()
    err_file.close()


if __name__ == "__main__":
    for dataset in ("Com2Sense", "CSQA", "CREAK"):
        DATASET = dataset
        main()
