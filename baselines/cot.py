import os
import re
import time
from tqdm import tqdm

import backoff
import google.generativeai as palm
import jsonlines
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

# from google.generativeai.types import safety_types

API_KEY = "api_key"
DATASET = "Com2Sense"

MAX_RETRY_COUNT = 4
TEMPERATURE = 0.3

safety_settings = [
    # {
    # "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
    # "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
    # },
    # {
    #     "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
    #     "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
    # },
    # {
    #     "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
    #     "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
    # },
    # {
    #     "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
    #     "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
    # },
    # {
    #     "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
    #     "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
    # },
    # {
    #     "category": safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
    #     "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
    # },
    # {
    #     "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
    #     "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
    # },
]


@backoff.on_exception(backoff.expo, ServiceUnavailable, max_tries=MAX_RETRY_COUNT)
@backoff.on_exception(backoff.expo, ResourceExhausted)
def prompt_func(model: str, prompt: str):
    return palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=TEMPERATURE,
        top_p=1,
        max_output_tokens=64,
        safety_settings=safety_settings,
    )


def main():
    log_file = open(f"time_{DATASET}_palm.log", "w")
    err_file = open(f"time_error_{DATASET}_palm.log", "w")

    palm.configure(api_key=API_KEY)

    model = "models/text-bison-001"
    with open(f"data/{DATASET}/few_shot_cot.txt", "r") as f:
        prompt_prefix = f.read()

    with jsonlines.open(f"data/{DATASET}/dev.Q.json", "r") as reader:
        data = list(reader)
    correct = 0
    start_time = time.time()
    for i, sample in tqdm(enumerate(data), file=(None if log_file is None else open(os.devnull, "w"))):
        sent = sample["Q"]
        label = sample["A"]
        prompt = f"{prompt_prefix}Q: {sent}\nA: "

        result, matched = None, None
        for _ in range(MAX_RETRY_COUNT):
            tmp = prompt_func(model, prompt)
            if tmp.result is not None:
                matched = re.search("this statement is (true|false)", tmp.result.lower())
                if matched:
                    result = tmp
                    break
            print("result:", tmp.result, file=err_file, flush=True)
            print("filteres:", tmp.filters, file=err_file, flush=True)
            print("Retrying...", file=err_file, flush=True)

        pred = matched is None or matched.groups()[0] == "true"
        if pred == label:
            correct += 1
        print(
            {
                "i": i,
                "result": pred == label,
                "label": label,
                "pred": pred,
                "Q": sent,
                "A": None if result is None else result.result,
            },
            file=log_file,
            flush=True,
        )
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
