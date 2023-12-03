import logging
import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from typing import Dict, List

import jsonlines
import torch
import transformers  # type: ignore
from tqdm import tqdm  # type: ignore
from transformers import RobertaForSequenceClassification, AutoTokenizer  # type: ignore
import config

from inference.inference import InferenceWrapper
from inference.verifier import NLIModel


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--device-id", default=0, type=int)
    parser.add_argument("--dataset-name", default="Com2Sense", type=str)
    parser.add_argument("-g", "--gen", action="store", default=0, type=int)
    parser.add_argument("--layer", action="store", default=0, type=int)
    parser.add_argument("-l", "--log", action="store", default=None)

    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.device_id}")
    args.data_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.Q.jsonl"
    args.out_filename = (
        f"./data/{args.dataset_name}/{args.gen}_gen/dev_layer{args.layer}.C.pkl"
    )
    args.G_filename = (
        f"./data/{args.dataset_name}/{args.gen}_gen/dev_layer{args.layer}.G.pkl"
    )

    return args


def setup_log(args: Namespace):
    if args.log is None:
        args.tqdm = True
        return
    args.log = f"./data/{args.dataset_name}/{args.gen}_gen/{args.log}"
    sys.stdout = sys.stderr = open(args.log, "a")
    args.tqdm = False

    logger = logging.getLogger(transformers.modeling_utils.__name__)
    logger.setLevel(logging.ERROR)


if __name__ == "__main__":
    parsed = parse_args()
    setup_log(parsed)

    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli").to(
        parsed.device
    )
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    InferenceWrapper.nli_model = NLIModel(model, tokenizer)

    with jsonlines.open(parsed.data_filename, "r") as reader:
        samples = list(reader)

    with open(parsed.G_filename, "rb") as fr:
        G_samples = pickle.load(fr)

    if os.path.exists(parsed.out_filename):
        with open(parsed.out_filename, "rb") as fr:
            orig_out_list = pickle.load(fr)
    else:
        orig_out_list = []
    out_stat = []

    # partial sample inference
    # samples = samples[:275]

    total_samples = len(samples)
    inferring_samples = len(G_samples) - len(orig_out_list)
    samples = samples[len(orig_out_list) :]
    if inferring_samples <= 0:
        print(f"No samples to infer", flush=True)
    else:
        print(f"Start processing '{parsed.data_filename}'", flush=True)
        final_layer = True
        out_list = orig_out_list
        for i, (sample, (_, G_list)) in enumerate(
            tqdm(
                zip(samples, G_samples),
                total=inferring_samples,
                disable=not parsed.tqdm,
            ),
            start=(len(out_list) + 1),
        ):
            inferred_correct_choices = []
            for Label, G in G_list:
                if G.size() == 1:
                    inferred_answer = (
                        1 if G["Q"].data["blf"][0] >= G["Q"].data["blf"][1] else -1
                    )
                elif G.size() > 1:
                    (
                        score_list,
                        correct_E_dict,
                        graph2sat,
                        belief,
                        consistency,
                    ) = InferenceWrapper.infer(G)
                    sum_score = sum([score[1] for score in score_list])
                    inferred_answer = 1 if sum_score >= 0 else -1
                else:
                    inferred_answer = 1

                if parsed.layer == 0:
                    if inferred_answer == 1:
                        inferred_correct_choices.append(Label)
                else:
                    Label_A, Label_B = Label
                    if Label_B == "" or inferred_answer == 1:
                        inferred_correct_choices.append(Label_A)
                    else:
                        inferred_correct_choices.append(Label_B)

            if len(inferred_correct_choices) != 1:
                final_layer = False

            # out_stat includes (answer key, list of inferred choices, number of choices)
            out_list.append(inferred_correct_choices)
            out_stat.append(
                (
                    sample["answerKey"],
                    inferred_correct_choices,
                    len(sample["question"]["choices"]),
                )
            )

            # Back up results
            if i % config.save_period == 0:
                with open(parsed.out_filename, "wb") as fw:
                    pickle.dump(out_list, fw)

        # save inferred choices result
        with open(parsed.out_filename, "wb") as fw:
            pickle.dump(out_list, fw)

        if final_layer:
            correct = 0
            for i, (key, inferred, total) in enumerate(out_stat, start=1):
                # print the result of each questions
                print(
                    f"Q_no: {i}/{total_samples}\t Correct: {key}\tInferred: {inferred}\t",
                    flush=True,
                )
                if key in inferred:
                    correct += 1

            print("Inferring Complete", flush=True)
            print(f"Total: {total_samples}", flush=True)
            print(f"Inferring: {inferring_samples}", flush=True)
            print(f"Correct: {correct}/{inferring_samples}", flush=True)
            print(f"Accuracy: {correct/inferring_samples}", flush=True)

        else:
            inferred_count_stat: Dict[int, List[float]] = dict()
            choices_left = 0
            total_choices = 0

            misinferred = 0
            correct_count = 0.0
            for i, (key, inferred, total) in enumerate(out_stat):
                # print the result of each questions
                print(
                    f"Q_no: {i}/{total_samples}\t Correct: {key}\tInferred: {inferred}\t",
                    flush=True,
                )

                inferred_count = len(inferred)
                if total not in inferred_count_stat.keys():
                    inferred_count_stat[total] = [0] * (total + 1)
                inferred_count_stat[total][inferred_count] += 1

                choices_left += inferred_count
                total_choices += total

                if key in inferred:
                    correct_count += 1 / inferred_count
                else:
                    misinferred += 1

            print("Finished Inferring; More layers needed", flush=True)
            print(f"Total: {total_samples}", flush=True)
            print(f"Inferring: {inferring_samples}", flush=True)

            # display inferred choice count
            for k in sorted(inferred_count_stat.keys()):
                print(
                    f"Total choices = {k}:\t {sum(inferred_count_stat[k])} questions",
                    flush=True,
                )
                for i, count in enumerate(inferred_count_stat[k]):
                    print(f"\tInferred {i} choices:\t {count} questions", flush=True)
            print(f"Misinferred: {misinferred}/{inferring_samples}", flush=True)
            print(
                f"Eliminated Choices: {total_choices - choices_left}/{total_choices}",
                flush=True,
            )
            print(
                f"Eliminated Ratio: {(total_choices - choices_left)/total_choices}",
                flush=True,
            )
            print(f"Remaining Choices: {choices_left}/{total_choices}", flush=True)
            print(f"Remaining Ratio: {choices_left/total_choices}", flush=True)
            print(f"Expected Correct: {correct_count}/{inferring_samples}", flush=True)
            print(f"Expected Accuracy: {correct_count/inferring_samples}", flush=True)

    # with open(args.out_filename, "wb") as f:
    #        pickle.dump(out_list, f)

    # print("Correct list:", flush=True)
    # print(sorted(tmp_correct), flush=True)
    # print("Wrong list:", flush=True)
    # print(sorted(tmp_wrong), flush=True)

    # print(f"Correct: {acc_result[0]}/{sum(acc_result)}", flush=True)
    # print(f"Accuracy: {acc_result[0]/sum(acc_result)}", flush=True)
