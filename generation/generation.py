import dataclasses
import itertools
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase
from treelib import Tree

from model_server.base import GenerationOutput
from model_server.client import ModelClient


@dataclass
class PromptConfig:
    max_new_tokens: int = 64
    do_sample: bool = True
    temperature: float = 1
    top_p: float = 1
    n: int = 3

    @classmethod
    def new(cls, max_new_tokens: int = 64, temperature: float = 1, top_p: float = 1, n: int = 3) -> Self:
        do_sample = temperature > 0
        if not do_sample:
            temperature = 1
        return cls(max_new_tokens, do_sample, temperature, top_p, n)

    def to_generation_config(self) -> GenerationConfig:
        kwargs = self.__dict__.copy()
        del kwargs["n"]
        return GenerationConfig(**kwargs)


class GenerationWrapper:
    """Given question, generate maieutic tree"""

    model_client: ModelClient
    tokenizer: PreTrainedTokenizerBase

    generation_prefix: Union[str, dict]
    belief_prefix: Union[str, dict]
    negation_prefix: str
    question_prefix: str

    generation_config: PromptConfig
    generation_config2: PromptConfig
    negation_config: PromptConfig
    belief_config: PromptConfig
    question_config: PromptConfig

    max_depth: int

    _true_index: int

    def __init__(
        self,
        server_config: Optional[str],
        generation_prefix: Union[str, dict],
        belief_prefix: Union[str, dict],
        negation_prefix: str,
        question_prefix: str,
        max_depth: int = 2,
    ):
        self.generation_prefix = generation_prefix
        self.belief_prefix = belief_prefix
        self.negation_prefix = negation_prefix
        self.question_prefix = question_prefix

        self.generation_config = PromptConfig.new(temperature=0.5)
        self.generation_config2 = PromptConfig.new(temperature=0.3, top_p=1, n=1)
        self.negation_config = PromptConfig.new(temperature=0, top_p=1, n=1)
        self.belief_config = PromptConfig.new(temperature=0.2, top_p=1, n=1)
        self.question_config = PromptConfig.new(temperature=0, top_p=1, n=1)
        self.max_depth = max_depth

        self.model_client = ModelClient(server_config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_client.info()["model_name"])

        self._true_index = self.tokenizer("true").input_ids[0]

    def create_maieutic_graph(self, Q: str, Q_tilde: str = None) -> Tree:
        # Initialize G
        G = Tree()
        if Q_tilde is None:
            Q, Q_tilde = self.prompt_Q_tilde(Q)
        G_blf, G_int = self.prompt_belief(Q, Q_tilde)
        G.create_node(Q, "Q", data={"E": Q, "E_tilde": Q_tilde, "blf": G_blf, "int": G_int})

        for depth in range(1, self.max_depth + 1):
            generation_config = self.generation_config if depth == 1 else self.generation_config2

            parents_to_generate_from = list(filter(lambda node: not node.data["int"], G.leaves()))
            for parent_node in parents_to_generate_from:
                new_E_T_list = self.prompt_E_T(parent_node.data["E"], dataclasses.replace(generation_config))
                new_E_T_tilde_list = self.prompt_tilde(new_E_T_list)
                beliefs = self.prompt_belief_list(new_E_T_list, new_E_T_tilde_list)
                for idx, (E_T, E_T_tilde, (E_blf, E_int)) in enumerate(zip(new_E_T_list, new_E_T_tilde_list, beliefs)):
                    node_identifier = f"{parent_node.identifier}T{idx}"
                    G.create_node(
                        E_T,
                        node_identifier,
                        parent=parent_node.identifier,
                        data={
                            "E": E_T,
                            "E_tilde": E_T_tilde,
                            "blf": E_blf,
                            "int": E_int,
                        },
                    )

                new_E_F_list = self.prompt_E_F(parent_node.data["E"], dataclasses.replace(generation_config))
                new_E_F_tilde_list = self.prompt_tilde(new_E_F_list)
                beliefs = self.prompt_belief_list(new_E_F_list, new_E_F_tilde_list)
                for idx, (E_F, E_F_tilde, (E_blf, E_int)) in enumerate(zip(new_E_F_list, new_E_F_tilde_list, beliefs)):
                    node_identifier = f"{parent_node.identifier}F{idx}"
                    G.create_node(
                        E_F,
                        node_identifier,
                        parent=parent_node.identifier,
                        data={
                            "E": E_F,
                            "E_tilde": E_F_tilde,
                            "blf": E_blf,
                            "int": E_int,
                        },
                    )

        # Remove branches that lead to logically not integral leaf nodes
        integral_leaf_nodes = [node.identifier for node in G.leaves() if node.data["int"]]
        paths_to_integral_leaves = [path for path in G.paths_to_leaves() if path[-1] in integral_leaf_nodes]
        nodes_not_to_remove = set(itertools.chain.from_iterable(paths_to_integral_leaves))

        nodes_before_removal = list(G.nodes.keys())
        for node in nodes_before_removal:
            if node in G and node not in nodes_not_to_remove:
                G.remove_node(node)

        return G

    def prompt_E_T(self, Q: str, generation_config: PromptConfig) -> List[str]:
        """Generate E_T from Q, using generation_config"""
        prompt_str = self.create_E_T_prompt(Q)
        num_Es_to_generate = generation_config.n
        E_T_list = []

        while len(E_T_list) < num_Es_to_generate:
            generation_config.n = num_Es_to_generate - len(E_T_list)
            response = self.model_client.generate(
                prompt_str, generation_config.to_generation_config(), generation_config.n
            )
            E_T_list.extend(GenerationWrapper.filter_generated_explanations(response))

        return E_T_list[:num_Es_to_generate]

    def prompt_E_F(self, Q: str, generation_config: PromptConfig) -> List[str]:
        """Generate E_F from Q, using generation_config"""
        prompt_str = self.create_E_F_prompt(Q)
        num_Es_to_generate = generation_config.n
        E_F_list = []

        while len(E_F_list) < num_Es_to_generate:
            generation_config.n = num_Es_to_generate - len(E_F_list)
            response = self.model_client.generate(
                prompt_str, generation_config.to_generation_config(), generation_config.n
            )
            E_F_list.extend(GenerationWrapper.filter_generated_explanations(response))

        return E_F_list[:num_Es_to_generate]

    def prompt_tilde(self, E: List[str]) -> List[str]:
        """
        Generate E_tilde given E
        """
        prompts = [self.create_negation_prompt(e) for e in E]
        response = self.model_client.batch_generate(
            prompts, self.negation_config.to_generation_config(), self.negation_config.n
        )
        E_tilde = [GenerationWrapper.filter_generated_explanations([r])[0] for r in response]
        return E_tilde

    def prompt_belief(self, Q: str, Q_tilde: str) -> Tuple[Tuple[Optional[float], Optional[float]], bool]:
        """
        Compute belief by comparing p(True|E) and p(True|not E).
        return p(True|E), p(True|not E), logical_integrity
        """
        probs = tuple(self.prompt_true_given_Q([Q, Q_tilde]))
        Q_E_Q_prob, Q_tilde_E_Q_tilde_prob = probs  # pylint: disable=unbalanced-tuple-unpacking

        if None not in probs:
            integrity = GenerationWrapper.logical_integrity(Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)
        else:
            integrity = False

        return probs, integrity

    def prompt_belief_list(
        self, Q: List[str], Q_tilde: List[str]
    ) -> List[Tuple[Tuple[Optional[float], Optional[float]], bool]]:
        tmp_probs = self.prompt_true_given_Q(Q + Q_tilde)
        Q_probs, Q_tilde_probs = tmp_probs[: len(Q)], tmp_probs[len(Q) :]
        results = tuple(
            (p, False if None in p else GenerationWrapper.logical_integrity(*p)) for p in zip(Q_probs, Q_tilde_probs)
        )
        return results

    def prompt_true_given_Q(self, Q: List[str], retry_count: int = 8) -> List[Optional[float]]:
        """
        Compute likelihood p(True|Q), and check whether max answer is True or False
        """
        prompts = [self.create_belief_prompt(q) for q in Q]
        generation_config = self.belief_config.to_generation_config()
        n = self.belief_config.n

        response = self.model_client.batch_generate(prompts, generation_config, n)
        true_given_Q = []
        for i, q in enumerate(Q):
            prob = self.retrieve_true_prob(response[n * i : n * (i + 1)])
            if prob is None:
                retry = self.model_client.generate(q, generation_config, retry_count)
                prob = self.retrieve_true_prob(retry)
            true_given_Q.append(prob)

        return true_given_Q

    def retrieve_true_prob(self, choices: List[GenerationOutput]) -> Optional[float]:
        """
        Given the model's choices, return likelihood of "true" in the generated text.
        Return None otherwise.
        """
        probs = []
        for choice in choices:
            generated_text = choice.text
            if re.search(". Therefore, the statement is (true|false).$", generated_text):
                token_index_list = self.tokenizer.batch_decode(self.tokenizer(generated_text).input_ids)
                true_or_false_index = len(token_index_list) - 3
                true_logprob = choice.scores[true_or_false_index][self._true_index]

                if not math.isinf(true_logprob):
                    probs.append(math.exp(true_logprob))
        if len(probs) == 0:
            return None
        return np.array(probs).mean()

    def prompt_Q_tilde(self, question: str, refine: bool = True) -> Tuple[str, str]:
        if refine and self.question_prefix is not None:
            prompt_str = self.create_Q_prompt(question)

            response = self.model_client.generate(
                prompt_str, self.generation_config.to_generation_config(), self.generation_config.n
            )
            refined_Q = GenerationWrapper.filter_generated_question(response)[0]
        else:
            refined_Q = question

        prompt_str = self.create_Q_tilde_prompt(refined_Q)

        response = self.model_client.generate(
            prompt_str, self.generation_config.to_generation_config(), self.generation_config.n
        )
        Q_tilde = GenerationWrapper.filter_generated_question(response)[0]

        return refined_Q, Q_tilde

    @staticmethod
    def filter_generated_explanations(explanations: List[GenerationOutput]) -> List[str]:
        # Extract string explanations
        filtered_explanations = [explanation.text.strip().strip("###").strip() for explanation in explanations]

        # Filter out empty string / those not ending with "."
        filtered_explanations = list(filter(lambda exp: len(exp) > 0 and exp.endswith("."), filtered_explanations))

        # Upper case the first letter
        filtered_explanations = [explanation[0].upper() + explanation[1:] for explanation in filtered_explanations]

        # If there's none left, just add the first one
        if len(filtered_explanations) == 0:
            filtered_explanations.append(explanations[0].text.strip())

        # Remove duplicates
        filtered_explanations = list(dict.fromkeys(filtered_explanations))

        return filtered_explanations

    @staticmethod
    def filter_generated_question(questions: List) -> List[str]:
        # Extract string questions
        filtered_questions = [question.text.strip() for question in questions]
        return filtered_questions

    @staticmethod
    def logical_integrity(prob1: float, prob2: float) -> bool:
        return abs(prob1 - prob2) > 0.45

    def create_negation_prompt(self, proposition: str) -> str:
        return f"{self.negation_prefix}\n" f"A: {proposition}\n" f"B: The statement is false."

    def create_belief_prompt(self, question: str) -> str:
        belief_prefix = self.belief_prefix
        question = question[:-1] + "?"
        return f"{belief_prefix}\n" f"Q: {question}\n" f"A:"

    def create_E_T_prompt(self, question: str) -> str:
        generation_prefix = self.generation_prefix
        question = question[:-1] + "?"
        return f"{generation_prefix}\n" f"Q: {question}\n" f"A: This statement is true, because"

    def create_E_F_prompt(self, question: str) -> str:
        generation_prefix = self.generation_prefix
        question = question[:-1] + "?"
        return f"{generation_prefix}\n" f"Q: {question}\n" f"A: This statement is false, because"

    def create_Q_prompt(self, question: str) -> str:
        question = question[:-1] + "."
        return f"{self.question_prefix}\n" f"Q: {question}\n" f"A: The statement is true."

    def create_Q_tilde_prompt(self, question: str) -> str:
        question = question[:-1] + "."
        return f"{self.negation_prefix}\n" f"Q: {question}\n" f"A: The statement is false."
