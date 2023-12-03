import dataclasses
import itertools
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import backoff
import google.generativeai as palm
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from google.generativeai.types import safety_types
from treelib import Tree

import config


class GenerationError(Exception):
    pass


@dataclass
class PromptConfig:
    model: str = config.model
    max_output_tokens: int = 64
    temperature: float = 1
    candidate_count: int = 3
    top_p: float = 1
    safety_settings = (
        {
            "category": safety_types.HarmCategory.HARM_CATEGORY_DANGEROUS,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": safety_types.HarmCategory.HARM_CATEGORY_SEXUAL,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": safety_types.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        },
        {
            "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        },
    )


class GenerationWrapper:
    """Given question, generate maieutic tree"""

    generation_prefix: str
    belief_prefix: str
    negation_prefix: str
    question_prefix: str

    generation_config: PromptConfig
    generation_config2: PromptConfig
    negation_config: PromptConfig
    belief_config: PromptConfig
    question_config: PromptConfig
    max_depth: int

    prompt_count: int = 0

    def __init__(
        self,
        generation_prefix: Union[str, dict],
        belief_prefix: Union[str, dict],
        negation_prefix: str,
        question_prefix: str,
        max_depth: int = config.max_depth,
    ):
        self.generation_prefix = generation_prefix
        self.belief_prefix = belief_prefix
        self.negation_prefix = negation_prefix
        self.question_prefix = question_prefix

        self.generation_config = PromptConfig(
            temperature=config.first_level_temp, candidate_count=config.first_level_candidate
        )
        self.generation_config2 = PromptConfig(
            temperature=config.second_level_temp, candidate_count=config.second_level_candidate
        )
        self.negation_config = PromptConfig(temperature=0, candidate_count=1)
        self.belief_config = PromptConfig(
            temperature=config.belief_temp,
            candidate_count=config.belief_candidate_count,
        )
        self.question_config = PromptConfig(temperature=0, candidate_count=1)
        self.max_depth = max_depth

    def create_maieutic_graph(self, Q: Optional[str], Q_tilde: Optional[str]):
        self.prompt_count = 0

        if Q is None:
            raise ValueError("The input question must not be None.")
        # Initialize G
        G = Tree()
        try:
            if Q_tilde is None:
                Q, Q_tilde = self.prompt_Q_tilde(Q)
            G_blf, G_int = self.prompt_belief(Q, Q_tilde)
        except GenerationError:
            return G
        G.create_node(Q, "Q", data={"E": Q, "E_tilde": Q_tilde, "blf": G_blf, "int": G_int})

        for depth in range(1, self.max_depth + 1):
            generation_config = self.generation_config if depth == 1 else self.generation_config2

            parents_to_generate_from = list(filter(lambda node: not node.data["int"], G.leaves()))
            for parent_node in parents_to_generate_from:
                try:
                    new_E_T_list = self.prompt_E_T(parent_node.data["E"], dataclasses.replace(generation_config))
                    new_E_T_tilde_list = [self.prompt_tilde(E_T) for E_T in new_E_T_list]

                    for idx, (E_T, E_T_tilde) in enumerate(zip(new_E_T_list, new_E_T_tilde_list)):
                        node_identifier = f"{parent_node.identifier}T{idx}"
                        E_blf, E_int = self.prompt_belief(E_T, E_T_tilde)
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
                    new_E_F_tilde_list = [self.prompt_tilde(E_F) for E_F in new_E_F_list]
                    for idx, (E_F, E_F_tilde) in enumerate(zip(new_E_F_list, new_E_F_tilde_list)):
                        node_identifier = f"{parent_node.identifier}F{idx}"
                        E_blf, E_int = self.prompt_belief(E_F, E_F_tilde)
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
                except GenerationError:
                    continue

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
        num_Es_to_generate = generation_config.candidate_count
        E_T_list = []

        while len(E_T_list) < num_Es_to_generate:
            generation_config.candidate_count = num_Es_to_generate - len(E_T_list)
            response = self.generate_text(prompt=prompt_str, **generation_config.__dict__)
            E_T_list.extend(GenerationWrapper.filter_generated_explanations(response.candidates))

        return E_T_list[:num_Es_to_generate]

    def prompt_E_F(self, Q: str, generation_config: PromptConfig) -> List[str]:
        """Generate E_F from Q, using generation_config"""
        prompt_str = self.create_E_F_prompt(Q)
        num_Es_to_generate = generation_config.candidate_count
        E_F_list = []

        while len(E_F_list) < num_Es_to_generate:
            generation_config.candidate_count = num_Es_to_generate - len(E_F_list)
            response = self.generate_text(prompt=prompt_str, **generation_config.__dict__)
            E_F_list.extend(GenerationWrapper.filter_generated_explanations(response.candidates))

        return E_F_list[:num_Es_to_generate]

    def prompt_tilde(self, E: str):
        """
        Generate E_tilde given E
        """
        prompt_str = self.create_negation_prompt(E)
        response = self.generate_text(prompt=prompt_str, **self.negation_config.__dict__)
        try:
            E_tilde = self.filter_generated_explanations(response.candidates)[0]
        except IndexError as e:
            print(response, flush=True)
            raise e

        return E_tilde

    def prompt_belief(self, Q: str, Q_tilde: str):
        """
        Compute belief by comparing p(True|E) and p(True|not E).
        return p(True|E), p(True|not E), logical_integrity
        """
        Q_E_Q_prob = self.prompt_true_given_Q(Q)
        Q_tilde_E_Q_tilde_prob = self.prompt_true_given_Q(Q_tilde)
        probs = (Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)

        if None not in probs:
            integrity = GenerationWrapper.logical_integrity(Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)
        else:
            integrity = False

        return probs, integrity

    def prompt_true_given_Q(self, Q: str):
        """
        Compute likelihood p(True|Q), and check whether max answer is True or False
        """
        prompt_str = self.create_belief_prompt(Q)

        response = self.generate_text(prompt=prompt_str, force=True, **self.belief_config.__dict__)
        true_given_Q = self.retrieve_true_prob(response.candidates)

        return true_given_Q

    def retrieve_true_prob(self, candidates: Dict[str, Any]):
        """
        Return the proportion of "true" in the generated candidates.
        Return None otherwise.
        """
        num_true = 0
        count = 0
        for candidate in candidates:
            matched = re.search(r"Therefore, the statement is (true|false)\.$", candidate["output"])
            if matched:
                count += 1
                if matched.groups()[0] == "true":
                    num_true += 1
        if count == 0:
            for candidate in candidates:
                print(candidate["output"], flush=True)
        return num_true / count if count != 0 else 0

    def prompt_Q_tilde(self, question: str, refine: bool = True) -> Tuple[str, str]:
        if refine and self.question_prefix is not None:
            prompt_str = self.create_Q_prompt(question)

            response = self.generate_text(prompt=prompt_str, **self.generation_config.__dict__)
            refined_Q = GenerationWrapper.filter_generated_question(response.candidates)[0]
        else:
            refined_Q = question

        prompt_str = self.create_Q_tilde_prompt(refined_Q)

        response = self.generate_text(prompt=prompt_str, **self.generation_config.__dict__)
        Q_tilde = GenerationWrapper.filter_generated_question(response.candidates)[0]

        return refined_Q, Q_tilde

    @staticmethod
    def filter_generated_explanations(explanations: List[Dict[str, Any]]):
        # Extract string explanations
        filtered_explanations = [explanation["output"].strip() for explanation in explanations]

        # Filter out empty string / those not ending with "."
        filtered_explanations = list(filter(lambda exp: len(exp) > 0 and exp.endswith("."), filtered_explanations))

        # Upper case the first letter
        filtered_explanations = [explanation[0].upper() + explanation[1:] for explanation in filtered_explanations]

        # If there's none left, just add the first one
        if len(filtered_explanations) == 0:
            filtered_explanations.append(explanations[0]["output"].strip())

        # Remove duplicates
        filtered_explanations = list(dict.fromkeys(filtered_explanations))

        return filtered_explanations

    @staticmethod
    def filter_generated_question(questions: List[Dict[str, Any]]):
        # Extract string questions
        filtered_questions = [question["output"].strip() for question in questions]
        return filtered_questions

    @staticmethod
    def logical_integrity(prob1: float, prob2: float):
        # return abs(prob1 - prob2) > 0.45
        return abs(prob1 - prob2) > config.belief_thresh

    def create_negation_prompt(self, proposition: str):
        return f"{self.negation_prefix}\n" f"A: {proposition}\n" f"B: The statement is false."

    def create_belief_prompt(self, question: str):
        belief_prefix = self.belief_prefix
        question = question[:-1] + "?"
        return f"{belief_prefix}\n" f"Q: {question}\n" f"A:"

    def create_E_T_prompt(self, question: str):
        generation_prefix = self.generation_prefix
        question = question[:-1] + "?"
        return f"{generation_prefix}\n" f"Q: {question}\n" f"A: This statement is true, because"

    def create_E_F_prompt(self, question: str):
        generation_prefix = self.generation_prefix
        question = question[:-1] + "?"
        return f"{generation_prefix}\n" f"Q: {question}\n" f"A: This statement is false, because"

    def create_Q_prompt(self, question: str):
        question = question[:-1] + "."
        return f"{self.question_prefix}\n" f"Q: {question}\n" f"A: The statement is true."

    def create_Q_tilde_prompt(self, question: str):
        question = question[:-1] + "."
        return f"{self.negation_prefix}\n" f"Q: {question}\n" f"A: The statement is false."

    @backoff.on_exception(backoff.expo, ServiceUnavailable, max_tries=4)
    @backoff.on_exception(backoff.expo, ResourceExhausted)
    def generate_text(
        self, prompt: str, candidate_count: int, force: bool = False, max_failure: int = 4, **kwargs
    ) -> palm.types.Completion:
        self.prompt_count += 1
        if force:
            candidates = []
            fail_count = 0
            while len(candidates) < candidate_count:
                response = palm.generate_text(prompt=prompt, candidate_count=candidate_count, **kwargs)
                valid = [
                    candidate
                    for candidate in response.candidates
                    if re.search(r"Therefore, the statement is (true|false)\.$", candidate["output"])
                ]
                if len(valid) == 0:
                    fail_count += 1
                if fail_count > max_failure:
                    raise GenerationError()
                candidates.extend(valid)
            response.candidates = candidates[:candidate_count]
        else:
            candidates = []
            fail_count = 0
            while len(candidates) < candidate_count:
                response = palm.generate_text(prompt=prompt, candidate_count=candidate_count, **kwargs)
                if len(response.candidates) == 0:
                    fail_count += 1
                if fail_count > max_failure:
                    raise GenerationError()
                candidates.extend(response.candidates)
            response.candidates = candidates[:candidate_count]
        return response
