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

import google.generativeai as palm  # type: ignore
import jsonlines
from treelib import Tree  # type: ignore

import config
from generation.generation import GenerationResult, GenerationWrapper
from generation.generation_config import retrieve_prompt_prefix


generator: GenerationWrapper
result_queue: Queue


@dataclass
class Record:
    id: int
    prompt_count: int
    time_taken: float
    depths: List[int]
    num_nodes: List[int]

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
    parser.add_argument("--layer", action="store", default=0, type=int)
    return parser.parse_args()


def pool_initializer(key_queue: Queue, resq: Queue, gen: GenerationWrapper):
    global generator, result_queue
    palm.configure(api_key=key_queue.get())
    result_queue = resq
    generator = gen


def process_sample(inp: Tuple[int, Tuple[Dict[str, Any], Optional[List[str]]]]):
    i, (sample, C_list) = inp

    start_time = time.time()
    question = sample["question"]
    answer_list = [
        (choice.get("label"), choice.get("text")) for choice in question.get("choices")
    ]
    graph_list: GenerationResult
    if C_list is None:
        graph_list = generator.create_single_choice_maieutic_graphs(
            question.get("stem"), answer_list
        )
    else:
        graph_list = generator.create_pairwise_maieutic_graphs(
            question.get("stem"), answer_list, C_list
        )

    end_time = time.time() - start_time

    result_queue.put((i, graph_list, generator.prompt_count, end_time))


def listener(
    result_queue: Queue,
    sample_count: int,
    total_sample_count: int,
    out_list: List[Tuple[int, Tuple[str, Tree]]],
    record_out_list: List[Record],
    log: Optional[Any],
    out_filename: str,
    record_filename: str,
):
    len_orig_out_list = len(out_list)
    for i in range(sample_count):
        sample_idx, graph_list, prompt_count, end_time = result_queue.get()
        bisect.insort(out_list, (sample_idx, graph_list))
        record = Record(
            sample_idx,
            prompt_count,
            end_time,
            *zip(*((G.depth(), G.size()) for (_, G) in graph_list)),
        )
        bisect.insort(record_out_list, record, key=lambda r: r.id)

        if log is not None:
            print(
                f"Sample: {i + len_orig_out_list + 1}/{total_sample_count}\tDepth: {sum(record.depths)}\tNodes: {sum(record.num_nodes)}\tTime taken (s): {record.time_taken:.4f}",
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
    # global generator

    args.data_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.Q.jsonl"
    args.out_filename = (
        f"./data/{args.dataset_name}/{args.gen}_gen/dev_layer{args.layer}.G.pkl"
    )
    args.record_filename = (
        f"./data/{args.dataset_name}/{args.gen}_gen/record_layer{args.layer}.jsonl"
    )
    if args.layer > 0:
        args.C_filename = (
            f"./data/{args.dataset_name}/{args.gen}_gen/dev_layer{args.layer - 1}.C.pkl"
        )

    prompt_prefix_dict = retrieve_prompt_prefix(args.dataset_name)
    generator = GenerationWrapper(
        generation_prefix=prompt_prefix_dict["abductive"],
        belief_prefix=prompt_prefix_dict["belief"],
        negation_prefix=prompt_prefix_dict["neg_explanation"],
        pairwise_belief_prefix=prompt_prefix_dict["pairwise_belief"],
        qa_negation_prefix=prompt_prefix_dict["qa_negation"],
        qa_prefix=prompt_prefix_dict["qa"],
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
    samples = [
        (i, sample)
        for i, sample in enumerate(samples)
        if i not in generated_sample_idxs
    ]

    if args.layer > 0:
        with open(args.C_filename, "rb") as f:
            C_samples = pickle.load(f)
        samples = [(i, (sample, C_samples[i])) for (i, sample) in samples]
    else:
        samples = [(i, (sample, None)) for i, sample in samples]

    print(f"Start processing '{args.data_filename}'", flush=True)
    start = time.time()

    key_queue: Queue[str] = Queue()
    result_queue: Queue[Tuple[int, List[Tuple[str, Tree]], float]] = Queue()
    for key in args.api_keys:
        key_queue.put(key)

    out_list = orig_out_list
    with Pool(
        len(args.api_keys), pool_initializer, (key_queue, result_queue, generator)
    ) as pool:
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
    print(f"Prompts per minute: {len(samples) / elapsed * 60}", flush=True)
    print()


def setup_log(args: Namespace):
    if args.log is None:
        return
    args.log = f"./data/{args.dataset_name}/{args.gen}_gen/{args.log}"
    log_file = open(args.log, "a")
    sys.stdout = log_file
    sys.stderr = log_file


if __name__ == "__main__":
    parsed = parse_args()

    parsed.tqdm_file = None if parsed.log is None else open(os.devnull, "w")

    for dataset in parsed.datasets:
        parsed.dataset_name = dataset
        setup_log(parsed)
        main(parsed)
