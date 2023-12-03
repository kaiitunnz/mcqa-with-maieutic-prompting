from typing import Final


PROMPT_CONFIGS: Final[dict] = {
    "CommonsenseQA": {
        "abductive": "./generation/prompts/CommonsenseQA/abductive.prompt.txt",
        "belief": "./generation/prompts/CommonsenseQA/belief.prompt.txt",
        "neg_explanation": "./generation/prompts/CommonsenseQA/neg_explanation.prompt.txt",
        "pairwise_belief": "./generation/prompts/CommonsenseQA/pairwise_belief.prompt.txt",
        "qa_negation": "./generation/prompts/CommonsenseQA/qa_negation.prompt.txt",
        "qa": "./generation/prompts/CommonsenseQA/qa.prompt.txt",
    },
    "ARCeasy": {
        "abductive": "./generation/prompts/ARC/abductive.prompt.txt",
        "belief": "./generation/prompts/ARC/belief.prompt.txt",
        "neg_explanation": "./generation/prompts/ARC/neg_explanation.prompt.txt",
        "pairwise_belief": "./generation/prompts/ARC/pairwise_belief.prompt.txt",
        "qa_negation": "./generation/prompts/ARC/qa_negation.prompt.txt",
        "qa": "./generation/prompts/ARC/qa.prompt.txt",
    },
    "ARCchallenge": {
        "abductive": "./generation/prompts/ARC/abductive.prompt.txt",
        "belief": "./generation/prompts/ARC/belief.prompt.txt",
        "neg_explanation": "./generation/prompts/ARC/neg_explanation.prompt.txt",
        "pairwise_belief": "./generation/prompts/ARC/pairwise_belief.prompt.txt",
        "qa_negation": "./generation/prompts/ARC/qa_negation.prompt.txt",
        "qa": "./generation/prompts/ARC/qa.prompt.txt",
    },
    "example": {
        "abductive": "./generation/prompts/example/abductive.prompt.txt",
        "belief": "./generation/prompts/example/belief.prompt.txt",
        "neg_explanation": "./generation/prompts/example/neg_explanation.prompt.txt",
        "pairwise_belief": "./generation/prompts/example/pairwise_belief.prompt.txt",
        "qa_negation": "./generation/prompts/example/qa_negation.prompt.txt",
        "qa": "./generation/prompts/example/qa.prompt.txt",
    },
}


def retrieve_prompt_prefix(dataset_name: str) -> dict:
    def open_file(data: dict):
        return {
            key: open_file(value)
            if isinstance(value, dict)
            else open(value, "r").read()
            for key, value in data.items()
        }

    filename_dict = PROMPT_CONFIGS[dataset_name]

    return open_file(filename_dict)
