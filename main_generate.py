import os
import pickle
import time
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List
from typing_extensions import Self

import google.generativeai as palm  # type: ignore
import jsonlines
from tqdm import tqdm  # type: ignore

import config
from generation.generation import GenerationResult, GenerationWrapper
from generation.generation_config import retrieve_prompt_prefix


@dataclass
class Record:
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
    parser.add_argument(
        "--api-key", action="store", default=config.palm_api_key, type=str
    )
    parser.add_argument("--datasets", action="extend", nargs="+", default=[], type=str)
    parser.add_argument("-l", "--log", action="store", default=None, type=str)
    parser.add_argument("-g", "--gen", action="store", default=0, type=int)
    parser.add_argument("--layer", action="store", default=0, type=int)
    return parser.parse_args()


def main(args: Namespace):
    palm.configure(api_key=args.api_key)

    args.data_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.Q.jsonl"
    args.out_filename = (
        f"./data/{args.dataset_name}/{args.gen}_gen/dev_layer{args.layer}.G.pkl"
    )
    args.record_filename = (
        f"./data/{args.dataset_name}/{args.gen}_gen/record_layer{args.layer}.jsonl"
    )
    if args.layer > 0:
        args.C_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev_layer{args.layer - 1}.Q.jsonl"

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

    if args.layer > 0:
        with open(args.C_filename, "rb") as f:
            C_samples = pickle.load(f)[len(orig_out_list) :]
    else:
        C_samples = [0] * (len(samples) - len(orig_out_list))

    total_samples = len(samples)
    samples = samples[len(orig_out_list) :]
    generating_samples = min(len(samples), len(C_samples))

    print(f"Start processing '{args.data_filename}'", flush=True)
    start = time.time()

    out_list = orig_out_list
    for sample_idx, (sample, choice_list) in enumerate(
        tqdm(
            zip(samples, C_samples),
            file=args.tqdm_file,
            total=(len(out_list) + generating_samples),
        ),
        start=(len(out_list) + 1),
    ):
        start_time = time.time()
        question = sample.get("question")
        answer_list = [
            (choice.get("label"), choice.get("text"))
            for choice in question.get("choices")
        ]
        graph_list: GenerationResult
        if args.layer > 0:
            graph_list = generator.create_single_choice_maieutic_graphs(
                question.get("stem"), answer_list
            )
        else:
            graph_list = generator.create_pairwise_maieutic_graphs(
                question.get("stem"), answer_list, choice_list
            )
        end_time = time.time() - start_time
        out_list.append(graph_list)
        record = Record(
            generator.prompt_count,
            end_time,
            *zip(*((G.depth(), G.size()) for (_, G) in graph_list)),
        )
        record_out_list.append(record)

        if args.log is not None:
            print(
                f"Sample: {sample_idx}/{total_samples}\tDepth: {sum(record.depths)}\tNodes: {sum(record.num_nodes)}\tTime taken (s): {record.time_taken:.4f}",
                flush=True,
            )

        if sample_idx % config.save_period == 0:  # backup save
            with open(args.out_filename, "wb") as f:
                pickle.dump(out_list, f)
            Record.dump(record_out_list, args.record_filename)

    elapsed = time.time() - start
    print(f"Finished processing {len(samples)} samples.", flush=True)
    print(f"Elapsed time: {elapsed}", flush=True)
    print(f"Prompts per minute: {len(samples) / elapsed * 60}", flush=True)
    print()

    with open(args.out_filename, "wb") as f:
        pickle.dump(out_list, f)
    Record.dump(record_out_list, args.record_filename)


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
