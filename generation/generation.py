import dataclasses
import itertools
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import backoff
import google.generativeai as palm  # type: ignore
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from google.generativeai.types import safety_types  # type: ignore
from treelib import Tree  # type: ignore

import config

SingleResult = List[Tuple[str, Tree]]
PairwiseResult = List[Tuple[Tuple[str, str], Tree]]
GenerationResult = Union[SingleResult, PairwiseResult]


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

    generation_prefix: Union[str, dict]
    belief_prefix: Union[str, dict]
    negation_prefix: str
    pairwise_belief_prefix: str
    qa_negation_prefix: str
    qa_prefix: str

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
        pairwise_belief_prefix: str,
        qa_negation_prefix: str,
        qa_prefix: str,
        max_depth: int = config.max_depth,
    ):
        self.generation_prefix = generation_prefix
        self.belief_prefix = belief_prefix
        self.negation_prefix = negation_prefix
        self.pairwise_belief_prefix = pairwise_belief_prefix
        self.qa_negation_prefix = qa_negation_prefix
        self.qa_prefix = qa_prefix

        self.generation_config = PromptConfig(
            temperature=config.first_level_temp,
            candidate_count=config.first_level_candidate,
        )
        self.generation_config2 = PromptConfig(
            temperature=config.second_level_temp,
            candidate_count=config.second_level_candidate,
        )
        self.negation_config = PromptConfig(temperature=0, candidate_count=1)
        self.belief_config = PromptConfig(
            temperature=config.belief_temp,
            candidate_count=config.belief_candidate_count,
        )
        self.question_config = PromptConfig(temperature=0, candidate_count=1)
        self.qa_statement_config = PromptConfig(temperature=0, candidate_count=1)
        self.max_depth = max_depth

    # step 1
    def create_single_choice_maieutic_graphs(
        self, Q: str, A_list: List[Tuple[str, str]]
    ) -> SingleResult:
        self.prompt_count = 0

        Tree_list = []
        for Label, Ans in A_list:
            try:
                QA_true = self.prompt_QA(Q, Ans)
                QA_false = self.prompt_QA_tilde(Q, Ans)
                Tree_list.append(
                    (Label, self.create_maieutic_graph(QA_true, QA_false, False))
                )
            except GenerationError:
                Tree_list.append((Label, Tree()))

        return Tree_list

    def create_pairwise_maieutic_graphs(
        self, Q: str, A_list: List[Tuple[str, str]], C_list: List[str]
    ) -> PairwiseResult:
        self.prompt_count = 0

        if len(C_list) > 0:
            valid_A_list = []
            for Label, Ans in A_list:
                if Label in C_list:
                    valid_A_list.append((Label, Ans))
        else:
            valid_A_list = A_list

        Tree_list = []
        finding_pair = None
        for A_tuple in valid_A_list:
            if finding_pair is None:
                finding_pair = A_tuple
            else:
                Label_A, Ans_A = finding_pair
                Label_B, Ans_B = A_tuple
                try:
                    Tree_list.append(
                        (
                            (Label_A, Label_B),
                            self.create_maieutic_graph(Q, (Ans_A, Ans_B), True),
                        )
                    )
                except GenerationError:
                    Tree_list.append(((Label_A, Label_B), Tree()))
                finding_pair = None

        if finding_pair is not None:
            # there is no pair; odd number of choices left
            Label, Ans = finding_pair
            Tree_list.append(((Label, ""), Tree()))

        return Tree_list

    def create_pairwise_abductive(self, Q: str, A: str, B: str) -> Tree:
        G = Tree()
        try:
            Q_A_blf, Q_A_int = self.prompt_pairwise_belief(Q, A, B)
        except GenerationError:
            return G
        G.create_node(
            Q,
            "Q",
            data={
                "E": self.prompt_QA(Q, A),
                "E_tilde": self.prompt_QA(Q, B),
                "blf": Q_A_blf,
                "int": Q_A_int,
            },
        )
        if not Q_A_int and G.depth() < self.max_depth:
            Q_node = G.leaves()[0]
            try:
                # debug
                # print(parent_node.data["E"], flush= True)

                new_E_A_list = self.prompt_E_A(
                    Q, A, B, dataclasses.replace(self.generation_config)
                )
                new_E_A_tilde_list = [self.prompt_tilde(E_T) for E_T in new_E_A_list]

                for idx, (E_A, E_A_tilde) in enumerate(
                    zip(new_E_A_list, new_E_A_tilde_list)
                ):
                    node_identifier = f"{Q_node.identifier}A{idx}"
                    E_blf, E_int = self.prompt_belief(E_A, E_A_tilde)
                    # debug
                    # print(f"int: {E_int}\t blf: {E_blf}\t E_T: {E_T}", flush= True)
                    G.create_node(
                        E_A,
                        node_identifier,
                        parent=Q_node.identifier,
                        data={
                            "E": E_A,
                            "E_tilde": E_A_tilde,
                            "blf": E_blf,
                            "int": E_int,
                        },
                    )

                new_E_B_list = self.prompt_E_A(
                    Q, B, A, dataclasses.replace(self.generation_config)
                )
                new_E_B_tilde_list = [self.prompt_tilde(E_B) for E_B in new_E_B_list]
                for idx, (E_B, E_B_tilde) in enumerate(
                    zip(new_E_B_list, new_E_B_tilde_list)
                ):
                    node_identifier = f"{Q_node.identifier}B{idx}"
                    E_blf, E_int = self.prompt_belief(E_B, E_B_tilde)
                    # debug
                    # print(f"int: {E_int}\t blf: {E_blf}\t E_F: {E_F}", flush= True)
                    G.create_node(
                        E_B,
                        node_identifier,
                        parent=Q_node.identifier,
                        data={
                            "E": E_B,
                            "E_tilde": E_B_tilde,
                            "blf": E_blf,
                            "int": E_int,
                        },
                    )
            except GenerationError:
                pass

        return G

    def create_maieutic_graph(
        self, Q: str, Q_tilde: Union[str, Tuple[str, str]], pairwise: bool
    ) -> Tree:
        if pairwise:
            assert isinstance(Q_tilde, tuple)
            G = self.create_pairwise_abductive(Q, *Q_tilde)
        else:
            assert isinstance(Q_tilde, str)
            G = Tree()
            try:
                G_blf, G_int = self.prompt_belief(Q, Q_tilde)
            except GenerationError:
                return G
            G.create_node(
                Q, "Q", data={"E": Q, "E_tilde": Q_tilde, "blf": G_blf, "int": G_int}
            )

        for depth in range(G.depth(), self.max_depth):
            generation_config = (
                self.generation_config if depth == 0 else self.generation_config2
            )

            parents_to_generate_from = list(
                filter(lambda node: not node.data["int"], G.leaves())
            )
            for parent_node in parents_to_generate_from:
                try:
                    # debug
                    # print(parent_node.data["E"], flush= True)

                    new_E_T_list = self.prompt_E_T(
                        parent_node.data["E"], dataclasses.replace(generation_config)
                    )
                    new_E_T_tilde_list = [
                        self.prompt_tilde(E_T) for E_T in new_E_T_list
                    ]

                    for idx, (E_T, E_T_tilde) in enumerate(
                        zip(new_E_T_list, new_E_T_tilde_list)
                    ):
                        node_identifier = f"{parent_node.identifier}T{idx}"
                        E_blf, E_int = self.prompt_belief(E_T, E_T_tilde, pairwise)
                        # debug
                        # print(f"int: {E_int}\t blf: {E_blf}\t E_T: {E_T}", flush= True)
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

                    new_E_F_list = self.prompt_E_F(
                        parent_node.data["E"], dataclasses.replace(generation_config)
                    )
                    new_E_F_tilde_list = [
                        self.prompt_tilde(E_F) for E_F in new_E_F_list
                    ]
                    for idx, (E_F, E_F_tilde) in enumerate(
                        zip(new_E_F_list, new_E_F_tilde_list)
                    ):
                        node_identifier = f"{parent_node.identifier}F{idx}"
                        E_blf, E_int = self.prompt_belief(E_F, E_F_tilde, pairwise)
                        # debug
                        # print(f"int: {E_int}\t blf: {E_blf}\t E_F: {E_F}", flush= True)
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
        integral_leaf_nodes = [
            node.identifier for node in G.leaves() if node.data["int"]
        ]
        paths_to_integral_leaves = [
            path for path in G.paths_to_leaves() if path[-1] in integral_leaf_nodes
        ]
        nodes_not_to_remove = set(
            itertools.chain.from_iterable(paths_to_integral_leaves)
        )

        nodes_before_removal = list(G.nodes.keys())
        for node in nodes_before_removal:
            if node in G and node not in nodes_not_to_remove:
                G.remove_node(node)

        return G

    def prompt_QA(self, Q: str, A: str) -> str:
        """Generate QA statement from Q and A, using generation_config"""
        prompt_str = self.create_QA_statement_prompt(Q, A)
        response = self.generate_text(
            prompt=prompt_str, **self.qa_statement_config.__dict__
        )
        try:
            QA_statement = GenerationWrapper.filter_generated_explanations(
                response.candidates
            )[0]
        except IndexError as e:
            print(response, flush=True)
            raise e

        return QA_statement

    def prompt_QA_tilde(self, Q: str, A: str) -> str:
        """Generate QA negation statement from Q and A, using generation_config"""
        prompt_str = self.create_QA_tilde_statement_prompt(Q, A)
        response = self.generate_text(
            prompt=prompt_str, **self.qa_statement_config.__dict__
        )
        try:
            QA_statement = GenerationWrapper.filter_generated_explanations(
                response.candidates
            )[0]
        except IndexError as e:
            print(response, flush=True)
            raise e

        return QA_statement

    def prompt_E_A(self, Q: str, A: str, B: str, generation_config: PromptConfig):
        prompt_str = self.create_E_A_prompt(Q, A, B)
        num_Es_to_generate = generation_config.candidate_count
        E_A_list: List[str] = []

        while len(E_A_list) < num_Es_to_generate:
            generation_config.candidate_count = num_Es_to_generate - len(E_A_list)
            response = self.generate_text(
                prompt=prompt_str, **generation_config.__dict__
            )
            E_A_list.extend(
                GenerationWrapper.filter_generated_explanations(response.candidates)
            )

        return E_A_list[:num_Es_to_generate]

    def prompt_E_T(self, Q: str, generation_config: PromptConfig) -> List[str]:
        """Generate E_T from Q, using generation_config"""
        prompt_str = self.create_E_T_prompt(Q)
        num_Es_to_generate = generation_config.candidate_count
        E_T_list: List[str] = []

        while len(E_T_list) < num_Es_to_generate:
            generation_config.candidate_count = num_Es_to_generate - len(E_T_list)
            response = self.generate_text(
                prompt=prompt_str, **generation_config.__dict__
            )
            E_T_list.extend(
                GenerationWrapper.filter_generated_explanations(response.candidates)
            )

        return E_T_list[:num_Es_to_generate]

    def prompt_E_F(self, Q: str, generation_config: PromptConfig) -> List[str]:
        """Generate E_F from Q, using generation_config"""
        prompt_str = self.create_E_F_prompt(Q)
        num_Es_to_generate = generation_config.candidate_count
        E_F_list: List[str] = []

        while len(E_F_list) < num_Es_to_generate:
            generation_config.candidate_count = num_Es_to_generate - len(E_F_list)
            response = self.generate_text(
                prompt=prompt_str, **generation_config.__dict__
            )
            E_F_list.extend(
                GenerationWrapper.filter_generated_explanations(response.candidates)
            )

        return E_F_list[:num_Es_to_generate]

    def prompt_tilde(self, E: str) -> str:
        """
        Generate E_tilde given E
        """
        prompt_str = self.create_negation_prompt(E)
        response = self.generate_text(
            prompt=prompt_str, **self.negation_config.__dict__
        )
        try:
            E_tilde = GenerationWrapper.filter_generated_explanations(
                response.candidates
            )[0]
        except IndexError as e:
            print(response, flush=True)
            raise e

        return E_tilde

    def prompt_pairwise_belief(
        self, Q: str, A: str, B: str
    ) -> Tuple[Tuple[Optional[float], Optional[float]], bool]:
        Q_A_prob = self.prompt_correct_given_Q(Q, A, B)
        Q_B_prob = self.prompt_correct_given_Q(Q, B, A)
        probs = (Q_A_prob, Q_B_prob)

        if not (Q_A_prob is None or Q_B_prob is None):
            integrity = GenerationWrapper.logical_integrity(Q_A_prob, Q_B_prob)
        else:
            integrity = False

        return probs, integrity

    def prompt_belief(
        self, Q: str, Q_tilde: str, pairwise: bool = False
    ) -> Tuple[Tuple[Optional[float], Optional[float]], bool]:
        """
        Compute belief by comparing p(True|E) and p(True|not E).
        return p(True|E), p(True|not E), logical_integrity
        """
        Q_E_Q_prob = self.prompt_true_given_Q(Q, pairwise)
        Q_tilde_E_Q_tilde_prob = self.prompt_true_given_Q(Q_tilde, pairwise)
        probs = (Q_E_Q_prob, Q_tilde_E_Q_tilde_prob)

        if None not in probs:
            integrity = GenerationWrapper.logical_integrity(
                Q_E_Q_prob, Q_tilde_E_Q_tilde_prob
            )
        else:
            integrity = False

        return probs, integrity

    def prompt_correct_given_Q(
        self, Q: str, correct_ans: str, incorrect_ans: str
    ) -> Optional[float]:
        prompt_str = self.create_pairwise_belief_prompt(Q, correct_ans, incorrect_ans)

        response = self.generate_text(
            prompt=prompt_str,
            pattern=r"^(more|less) correct",
            **self.belief_config.__dict__,
        )
        true_given_Q = self.retrieve_correct_prob(response.candidates)

        return true_given_Q

    def prompt_true_given_Q(self, Q: str, pairwise: bool):
        """
        Compute likelihood p(True|Q), and check whether max answer is True or False
        """
        prompt_str = self.create_belief_prompt(Q, pairwise)

        response = self.generate_text(
            prompt=prompt_str,
            pattern=r"Therefore, the statement is (true|false)\.$",
            **self.belief_config.__dict__,
        )
        true_given_Q = self.retrieve_true_prob(response.candidates)

        return true_given_Q

    def retrieve_correct_prob(self, candidates: Dict) -> float:
        num_correct = 0
        count = 0
        for candidate in candidates:
            matched = re.search(r"^(more|less) correct", candidate["output"])
            if matched:
                count += 1
                if matched.groups()[0] == "more":
                    num_correct += 1
        if count == 0:
            for candidate in candidates:
                print(candidate["output"], flush=True)
        return num_correct / count if count != 0 else 0

    def retrieve_true_prob(self, candidates: Dict) -> float:
        """
        Return the proportion of "true" in the generated candidates.
        Return None otherwise.
        """
        num_true = 0
        count = 0
        for candidate in candidates:
            matched = re.search(
                r"Therefore, the statement is (true|false)\.$", candidate["output"]
            )
            if matched:
                count += 1
                if matched.groups()[0] == "true":
                    num_true += 1
        if count == 0:
            for candidate in candidates:
                print(candidate["output"], flush=True)
        return num_true / count if count != 0 else 0

    @staticmethod
    def filter_generated_explanations(explanations: List[Dict[str, Any]]):
        # Extract string explanations
        filtered_explanations = [
            explanation["output"].strip() for explanation in explanations
        ]

        # Filter out empty string / those not ending with "."
        filtered_explanations = list(
            filter(
                lambda exp: len(exp) > 0 and exp.endswith("."), filtered_explanations
            )
        )

        # Upper case the first letter
        filtered_explanations = [
            explanation[0].upper() + explanation[1:]
            for explanation in filtered_explanations
        ]

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

    def create_QA_statement_prompt(self, question: str, answer: str):
        return f"{self.qa_prefix}\n" f"Q: {question}\n" f"A: {answer}\n" f"Statement:"

    def create_QA_tilde_statement_prompt(self, question: str, answer: str):
        return (
            f"{self.qa_negation_prefix}\n"
            f"Q: {question}\n"
            f"A: {answer}\n"
            f"Statement:"
        )

    def create_negation_prompt(self, proposition: str):
        return (
            f"{self.negation_prefix}\n"
            f"Q: {proposition}\n"
            f"A: The statement is false."
        )

    def create_pairwise_belief_prompt(self, question: str, A: str, B: str) -> str:
        return (
            f"{self.pairwise_belief_prefix}\n"
            f"Q: {question}\n"
            f"A: {A}\n"
            f"B: {B}\n"
            f"Answer: A is"
        )

    def create_belief_prompt(self, question: str, pairwise: bool):
        belief_prefix = str(self.belief_prefix)
        question = question[:-1] + "?"
        return f"{belief_prefix}\n" f"Q: {question}\n" f"A:"

    def create_E_A_prompt(
        self, question: str, correct_ans: str, incorrect_ans: str
    ) -> str:
        return (
            f"{self.pairwise_belief_prefix}\n"
            f"Q: {question}\n"
            f"A: {correct_ans}\n"
            f"B: {incorrect_ans}\n"
            f"Answer: A is more correct because"
        )

    def create_E_T_prompt(self, question: str):
        generation_prefix = str(self.generation_prefix)
        question = question[:-1] + "?"
        return (
            f"{generation_prefix}\n"
            f"Q: {question}\n"
            f"A: This statement is true, because"
        )

    def create_E_F_prompt(self, question: str):
        generation_prefix = str(self.generation_prefix)
        question = question[:-1] + "?"
        return (
            f"{generation_prefix}\n"
            f"Q: {question}\n"
            f"A: This statement is false, because"
        )

    def create_Q_tilde_prompt(self, question: str):
        question = question[:-1] + "."
        return (
            f"{self.negation_prefix}\n" f"Q: {question}\n" f"A: The statement is false."
        )

    @backoff.on_exception(backoff.expo, ServiceUnavailable, max_tries=4)
    @backoff.on_exception(backoff.expo, ResourceExhausted)
    def generate_text(
        self,
        prompt: str,
        candidate_count: int,
        pattern: Optional[str] = None,
        max_failure: int = 4,
        **kwargs,
    ) -> palm.types.Completion:
        self.prompt_count += 1
        if pattern is not None:
            candidates: List[Any] = []
            fail_count = 0
            while len(candidates) < candidate_count:
                response = palm.generate_text(
                    prompt=prompt, candidate_count=candidate_count, **kwargs
                )
                valid = [
                    candidate
                    for candidate in response.candidates
                    if re.search(pattern, candidate["output"])
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
                response = palm.generate_text(
                    prompt=prompt, candidate_count=candidate_count, **kwargs
                )
                if len(response.candidates) == 0:
                    fail_count += 1
                if fail_count > max_failure:
                    raise GenerationError()
                candidates.extend(response.candidates)
            response.candidates = candidates[:candidate_count]
        return response
