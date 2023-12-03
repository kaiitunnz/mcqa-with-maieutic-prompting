import re
import sys
import time
from argparse import ArgumentParser, Namespace

import jsonlines
from tqdm import tqdm  # type: ignore

import palm

API_KEY = "api_key"


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset-name", required=True, type=str)
    parser.add_argument("-l", "--log", default=None, type=str)

    return parser.parse_args()


def main(args: Namespace):
    assert args.dataset_name is not None

    palm.configure(API_KEY)

    with open(f"./data/{args.dataset_name}/cot_prefix.txt", "r") as f:
        prompt_prefix = f.read()

    with jsonlines.open(f"./data/{args.dataset_name}/dev.Q.jsonl", "r") as reader:
        data = list(reader)

    correct = 0
    start_time = time.time()
    for i, sample in tqdm(enumerate(data), disable=args.log is not None):
        question = sample["question"]["stem"]
        choices = sorted(
            (choice["label"].lower(), choice["text"])
            for choice in sample["question"]["choices"]
        )
        possible_choices = "|".join(label for label, _ in choices)
        label = sample["answerKey"].lower()
        prompt = (
            f"{prompt_prefix}\n"
            f"Q: {question}\n"
            f"Answer Choices: {' '.join((f'({label}) {text}' for label, text in choices))}\n"
            "A:"
        )

        result, matched = None, None
        for _ in range(palm.MAX_RETRY_COUNT):
            tmp = palm.generate_text(prompt)
            if tmp.result is not None:
                matched = re.search(
                    f"the answer is \(({possible_choices.lower()})\).$",
                    tmp.result.lower(),
                )
                if matched:
                    result = tmp
                    break
            print("result:", tmp.result, flush=True)
            print("filteres:", tmp.filters, flush=True)
            print("Retrying...", flush=True)

        pred = choices[0][0] if matched is None else matched.groups()[0]
        if pred == label:
            correct += 1
        print(
            {
                "i": i,
                "result": pred == label,
                "label": label,
                "pred": pred,
                "question": question,
                "choices": choices,
                "answer": None if result is None else result.result,
            },
            flush=True,
        )
    elapsed_time = time.time() - start_time

    print(f"Correct: {correct}/{len(data)}", flush=True)
    print(f"Accuracy: {correct/len(data)}", flush=True)
    print(f"Finished in: {elapsed_time} seconds", flush=True)
    print(f"Samples per minute: {len(data) / (elapsed_time) * 60}", flush=True)


if __name__ == "__main__":
    args = parse_args()
    if args.log is not None:
        sys.stdout = sys.stderr = open(args.log, "a")

    main(args)
