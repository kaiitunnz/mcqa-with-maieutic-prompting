import bisect
import os
import pickle
import time
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from multiprocessing import Pool, Process, Queue
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Self

import google.generativeai as palm
import jsonlines
from treelib import Tree

import config
from generation.generation import GenerationWrapper
from generation.generation_config import retrieve_prompt_prefix

generator: GenerationWrapper
result_queue: Queue


@dataclass
class Record:
    id: int
    prompt_count: int
    time_taken: float
    depth: int
    num_nodes: int

    @classmethod
    def load(cls, fname: str) -> List[Self]:
        with jsonlines.open(fname, "r") as reader:
            return [cls(**line) for line in reader]

    @classmethod
    def dump(cls, records: List[Self], fname: str):
        with jsonlines.open(fname, "w") as writer:
            writer.write_all(r.__dict__ for r in records)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--api-keys", action="extend", nargs="+", default=[], type=str)
    parser.add_argument("--datasets", action="extend", nargs="+", default=[], type=str)
    parser.add_argument("-l", "--log", action="store", default=None, type=str)
    parser.add_argument("-g", "--gen", action="store", default=0, type=int)
    return parser.parse_args()


def pool_initializer(key_queue: Queue, resq: Queue, gen: GenerationWrapper):
    global generator, result_queue  # pylint: disable=global-statement
    palm.configure(api_key=key_queue.get())
    result_queue = resq
    generator = gen


def process_sample(inp: Tuple[int, Dict[str, Any]]):
    i, sample = inp

    start_time = time.time()
    G = generator.create_maieutic_graph(sample.get("Q"), sample.get("Q_tilde"))
    end_time = time.time() - start_time

    result_queue.put((i, G, generator.prompt_count, end_time))


def listener(
    result_queue: Queue,
    sample_count: int,
    total_sample_count: int,
    out_list: List[Tuple[int, Tree]],
    record_out_list: List[Record],
    log: Optional[Any],
    out_filename: str,
    record_filename: str,
):
    len_orig_out_list = len(out_list)
    for i in range(sample_count):
        sample_idx, G, prompt_count, end_time = result_queue.get()
        bisect.insort(out_list, (sample_idx, G))
        record = Record(sample_idx, prompt_count, end_time, G.depth(), G.size())
        bisect.insort(record_out_list, record, key=lambda r: r.id)

        if log is not None:
            print(
                f"Sample: {i + len_orig_out_list + 1}/{total_sample_count}\tDepth: {record.depth}\tNodes: {record.num_nodes}\tTime taken (s): {record.time_taken:.4f}",
                flush=True,
            )

        if i % config.save_period == 0:  # backup save
            with open(out_filename, "wb") as f:
                pickle.dump(out_list, f)
            Record.dump(record_out_list, record_filename)

    with open(out_filename, "wb") as f:
        pickle.dump(out_list, f)
    Record.dump(record_out_list, record_filename)


def main(args: Namespace):
    args.data_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.Q.json"
    args.out_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.G.pkl"
    args.record_filename = f"./data/{args.dataset_name}/{args.gen}_gen/record.jsonl"

    prompt_prefix_dict = retrieve_prompt_prefix(args.dataset_name)
    generator = GenerationWrapper(
        prompt_prefix_dict["abductive"],
        prompt_prefix_dict["belief"],
        prompt_prefix_dict["negation"],
        prompt_prefix_dict["question"],
        max_depth=2,
    )

    if os.path.exists(args.out_filename):
        with open(args.out_filename, "rb") as f:
            orig_out_list = pickle.load(f)
    else:
        orig_out_list = []

    if os.path.exists(args.record_filename):
        record_out_list = Record.load(args.record_filename)
    else:
        record_out_list = []

    with jsonlines.open(args.data_filename, "r") as f:
        samples = list(f)

    total_samples = len(samples)
    generated_sample_idxs = set(sample[0] for sample in orig_out_list)
    samples = [(i, sample) for i, sample in enumerate(samples) if i not in generated_sample_idxs]

    print(f"Start processing '{args.data_filename}'", flush=True)
    start = time.time()

    key_queue = Queue()
    result_queue = Queue()
    for key in args.api_keys:
        key_queue.put(key)

    out_list = orig_out_list
    with Pool(len(args.api_keys), pool_initializer, (key_queue, result_queue, generator)) as pool:
        listener_process = Process(
            target=listener,
            args=(
                result_queue,
                len(samples),
                total_samples,
                out_list,
                record_out_list,
                args.log,
                args.out_filename,
                args.record_filename,
            ),
        )
        listener_process.start()
        result = pool.map_async(process_sample, samples)
        result.get()
        elapsed = time.time() - start
        listener_process.join()

    print(f"Finished processing {len(samples)} samples.", flush=True)
    print(f"Elapsed time: {elapsed}", flush=True)
    print(f"Prompts per minute (parallel): {len(samples) / elapsed * 60}", flush=True)
    record_out_list = Record.load(args.record_filename)
    total_time_taken = sum(record.time_taken for record in record_out_list)
    print(f"Prompts per minute (sequential): {len(samples) / total_time_taken * 60}", flush=True)
    print()


if __name__ == "__main__":
    parsed = parse_args()

    if parsed.log is None:
        parsed.tqdm_file = None
    else:
        log_file = open(parsed.log, "a")
        sys.stdout = log_file
        sys.stderr = log_file
        parsed.tqdm_file = open(os.devnull, "w")

    for dataset in parsed.datasets:
        parsed.dataset_name = dataset
        main(parsed)
