import os
import pickle
import sys
import time
from argparse import ArgumentParser, Namespace

import jsonlines
from tqdm import tqdm

from generation.generation import GenerationWrapper
from generation.generation_config import retrieve_prompt_prefix


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--datasets", action="extend", nargs="+", default=[], type=str)
    parser.add_argument("-l", "--log", action="store", default=None, type=str)
    parser.add_argument("-g", "--gen", action="store", default=0, type=int)
    parser.add_argument("--save-period", action="store", default=1, type=int)
    parser.add_argument("--server-config", action="store", default=None)
    return parser.parse_args()


def main(args: Namespace):
    args.data_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.Q.json"
    args.out_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.G.pkl"

    prompt_prefix_dict = retrieve_prompt_prefix(args.dataset_name)
    generator = GenerationWrapper(
        args.server_config,
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

    with jsonlines.open(args.data_filename, "r") as f:
        samples = list(f)

    total_samples = len(samples)
    samples = samples[len(orig_out_list) :]

    print(f"Start processing '{args.data_filename}'", flush=True)
    start = time.time()

    out_list = orig_out_list
    for sample_idx, sample in enumerate(tqdm(samples, file=args.tqdm_file), start=(len(out_list) + 1)):
        start_time = time.time()
        G = generator.create_maieutic_graph(sample.get("Q"), sample.get("Q_tilde"))
        end_time = time.time() - start_time
        out_list.append(G)

        if args.log is not None:
            print(
                f"Sample: {sample_idx}/{total_samples}\tDepth: {G.depth()}\tNodes: {G.size()}\tTime taken (s): {end_time:.4f}",
                flush=True,
            )

        if sample_idx % args.save_period == 0:  # backup save
            with open(args.out_filename, "wb") as f:
                pickle.dump(out_list, f)

    elapsed = time.time() - start
    print(f"Finished processing {len(samples)} samples.", flush=True)
    print(f"Elapsed time: {elapsed}", flush=True)
    print(f"Prompts per minute: {len(samples) / elapsed * 60}", flush=True)
    print()

    with open(args.out_filename, "wb") as f:
        pickle.dump(out_list, f)


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
