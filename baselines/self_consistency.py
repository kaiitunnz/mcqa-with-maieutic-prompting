import re
import sys
import time
from argparse import ArgumentParser, Namespace
from collections import Counter
from multiprocessing import Pool, Process, Queue
from typing import Any, Dict, List, Tuple

import google.generativeai as genai  # type: ignore
import jsonlines

import palm

MAX_CANDIDATE_COUNT = 8
NUM_SAMPLES = 40
TEMPERATURE = 0.5

result_queue: Queue
prompt_prefix: str


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--api-keys", action="extend", nargs="+", default=[], type=str)
    parser.add_argument("--dataset-name", action="store", type=str)
    parser.add_argument("-l", "--log", action="store", default=None, type=str)
    return parser.parse_args()


def generate_text(
    prompt: str,
    candidate_count: int,
    temperature: float,
    pattern: str,
    max_failure: int = 4,
) -> genai.types.Completion:
    candidates: List[Any] = []
    fail_count = 0
    while len(candidates) < candidate_count:
        response = palm.generate_text(
            prompt=prompt,
            candidate_count=min(candidate_count, MAX_CANDIDATE_COUNT),
            temperature=temperature,
        )
        valid = [
            candidate
            for candidate in response.candidates
            if re.search(pattern, candidate["output"])
        ]
        if len(valid) == 0:
            fail_count += 1
        if fail_count > max_failure:
            return response
        candidates.extend(valid)
    response.candidates = candidates[:candidate_count]
    return response


def pool_initializer(key_queue: Queue, resq: Queue):
    global result_queue
    palm.configure(api_key=key_queue.get())
    result_queue = resq


def process_sample(inp: Tuple[int, Dict[str, Any]]):
    i, sample = inp

    start_time = time.time()
    question: Dict[str, Any] = sample["question"]
    stem = sample["question"]["stem"]
    choices = sorted(
        (choice["label"].lower(), choice["text"])
        for choice in sample["question"]["choices"]
    )
    possible_choices = "|".join(label for label, _ in choices)
    label = sample["answerKey"].lower()
    prompt = (
        f"{prompt_prefix}\n"
        f"Q: {stem}\n"
        f"Answer Choices: {' '.join((f'({label}) {text}' for label, text in choices))}\n"
        "A:"
    )

    result = None
    pattern = f"the answer is \(({possible_choices.lower()})\)"
    tmp = generate_text(
        prompt,
        candidate_count=NUM_SAMPLES,
        temperature=TEMPERATURE,
        pattern=pattern,
    )
    result = []
    for candidate in tmp.candidates:
        matched = re.search(pattern, candidate["output"])
        if matched:
            result.append(matched.groups()[0])
    most_common = Counter(result).most_common(1)
    pred = None if len(most_common) == 0 else most_common[0][0]

    end_time = time.time() - start_time

    result_queue.put((i, question, choices, pred, label, end_time))


def listener(
    result_queue: Queue,
    sample_count: int,
):
    correct = 0
    total_time = 0
    for _ in range(sample_count):
        sample_idx, question, choices, pred, label, end_time = result_queue.get()
        total_time += end_time
        if pred == label:
            correct += 1
        print(
            {
                "i": sample_idx,
                "result": pred == label,
                "label": label,
                "pred": pred,
                "question": question,
                "choices": choices,
            },
            flush=True,
        )

    print(f"Correct: {correct}/{sample_count}", flush=True)
    print(f"Accuracy: {correct/sample_count}", flush=True)
    print(f"Finished in: {total_time} seconds", flush=True)
    print(f"Samples per minute: {sample_count / (total_time) * 60}", flush=True)


def main(args: Namespace):
    global prompt_prefix

    assert args.dataset_name is not None

    with open(f"./data/{args.dataset_name}/cot_prefix.txt", "r") as f:
        prompt_prefix = f.read()

    with jsonlines.open(f"./data/{args.dataset_name}/dev.Q.jsonl", "r") as reader:
        samples = list(reader)

    key_queue: Queue[str] = Queue()
    result_queue: Queue[Tuple[int, str, Dict[str, str], str, str, float]] = Queue()
    for key in args.api_keys:
        key_queue.put(key)

    with Pool(len(args.api_keys), pool_initializer, (key_queue, result_queue)) as pool:
        listener_process = Process(
            target=listener,
            args=(
                result_queue,
                len(samples),
            ),
        )
        listener_process.start()
        result = pool.map_async(process_sample, enumerate(samples))
        result.get()
        listener_process.join()


if __name__ == "__main__":
    args = parse_args()
    if args.log is not None:
        sys.stdout = sys.stderr = open(args.log, "a")

    main(args)
