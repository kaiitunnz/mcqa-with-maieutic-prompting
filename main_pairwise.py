import os
import pickle
import time
import sys
from argparse import ArgumentParser, Namespace
from typing import List

import google.generativeai as palm
import jsonlines
from tqdm import tqdm

import config
from generation.generation import GenerationWrapper
from generation.generation_config import retrieve_prompt_prefix

import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
from treelib import Tree

from inference.inference import InferenceWrapper
from inference.verifier import NLIModel


def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument("--device_id", default=0, type=int)
    args.add_argument("--dataset_name", default="Com2Sense", type=str)
    args.add_argument("-g", "--gen", action="store", default=0, type=int)
    args.add_argument("--layer", action="store", default=0, type=int)

    args = args.parse_args()
    args.device = torch.device(f"cuda:{args.device_id}")
    args.data_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.Q.json"
    args.G_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.G.pkl"
    if (args.gen > 0):
        args.data_filename = f"./data/{args.dataset_name}/{args.gen}_gen/dev.Q.jsonl"

    return args


if __name__ == "__main__":
    args = parse_args()
    model = RobertaForSequenceClassification.from_pretrained("roberta-large-mnli").to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    InferenceWrapper.nli_model = NLIModel(model, tokenizer)

    with jsonlines.open(args.data_filename, "r") as f:
        samples = list(f)

    with open(args.G_filename, "rb") as f:
        G_samples: list[list[(str, Tree)]] = pickle.load(f)

    acc_result = [0, 0]  # [correct, incorrect]
    # tmp_correct = []
    # tmp_wrong = []
    for i, (sample, G_list) in enumerate(tqdm(zip(samples, G_samples), total=len(samples))):
        inferred_correct_choices = []
        for (Label, G) in G_list:
            if G.size() == 1:
                inferred_answer = 1 if G["Q"].data["blf"][0] >= G["Q"].data["blf"][1] else -1
            elif G.size() > 1:
                score_list, correct_E_dict, graph2sat, belief, consistency = InferenceWrapper.infer(G)
                sum_score = sum([score[1] for score in score_list])
                inferred_answer = 1 if sum_score >= 0 else -1
            else:
                inferred_answer = 1

            if inferred_answer == 1:
                inferred_correct_choices.append(Label)


        # Record results
        # gt_answer = 1 if sample["A"] else -1
        # acc_result[0 if inferred_answer == gt_answer else 1] += 1

        # if inferred_answer == gt_answer:
        #     tmp_correct.append(G.size())
        # else:
        #     tmp_wrong.append(G.size())

        print(
            {
                "i": i,
                "inferred answer": inferred_correct_choices,
                "correct answer": sample["answerKey"],
            },
        )
        print(str(G))

    # print("Correct list:")
    # print(sorted(tmp_correct))
    # print("Wrong list:")
    # print(sorted(tmp_wrong))

    #print(f"Correct: {acc_result[0]}/{sum(acc_result)}")
    #print(f"Accuracy: {acc_result[0]/sum(acc_result)}")
