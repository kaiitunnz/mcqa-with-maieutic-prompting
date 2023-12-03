import json
import math
import pickle
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional
from typing_extensions import Self

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig


class ServerConfig(NamedTuple):
    address: str
    port: int
    log: Optional[str]
    verbose: bool

    @classmethod
    def new(cls, config_file: str) -> Self:
        with open(config_file, "r") as f:
            config = json.load(f)
        return cls(**config)


class ModelInput(NamedTuple):
    prompts: List[str]
    generation_config: GenerationConfig


class BaseRequest:
    @classmethod
    def new(cls, raw: Dict[str, Any]) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def get_model_input(self) -> ModelInput:
        raise NotImplementedError()


@dataclass
class SingleRequest(BaseRequest):
    prompt: str
    generation_config: GenerationConfig
    n: int
    path: str = "/single"

    @classmethod
    def new(cls, raw: Dict[str, Any]) -> Self:
        prompt = raw.get("prompt")
        generation_config = raw.get("parameters")
        n = raw.get("n")
        if prompt is None or generation_config is None or n is None:
            raise ValueError()
        return cls(prompt, GenerationConfig(**generation_config), n)

    def to_dict(self) -> Dict[str, Any]:
        return {"prompt": self.prompt, "parameters": self.generation_config.to_dict(), "n": self.n}

    def get_model_input(self) -> ModelInput:
        return ModelInput([self.prompt] * self.n, self.generation_config)


@dataclass
class BatchRequest(BaseRequest):
    prompts: List[str]
    generation_config: GenerationConfig
    n: int
    path: str = "/batch"

    @classmethod
    def new(cls, raw: Dict[str, Any]) -> Self:
        prompts = raw.get("prompts")
        generation_config = raw.get("parameters")
        n = raw.get("n")
        if prompts is None or generation_config is None or n is None:
            raise ValueError()
        return cls(prompts, GenerationConfig(**generation_config), n)

    def to_dict(self) -> Dict[str, Any]:
        return {"prompts": self.prompts, "parameters": self.generation_config.to_dict(), "n": self.n}

    def get_model_input(self) -> ModelInput:
        prompts = [prompt for prompt in self.prompts for _ in range(self.n)]
        return ModelInput(prompts, self.generation_config)


@dataclass
class GenerationOutput:
    text: str
    scores: np.ndarray

    def to_json(self) -> Dict[str, Any]:
        json_dict = self.__dict__.copy()
        json_dict["scores"] = pickle.dumps(self.scores).decode("latin-1")
        return json_dict

    @classmethod
    def from_json(cls, json_obj: Dict[str, Any]) -> Optional[Self]:
        text = json_obj["text"]
        scores = json_obj["scores"]
        if text is None or scores is None:
            raise ValueError()
        return cls(text=text, scores=pickle.loads(scores.encode("latin-1")))

    def __repr__(self) -> str:
        return str(self.__dict__.copy())


class ServerResponse:
    outputs: List[GenerationOutput]

    def __init__(self, outputs: List[GenerationOutput]):
        self.outputs = outputs

    @classmethod
    def new(cls, texts: List[str], scores: List[np.ndarray]) -> Self:
        assert len(texts) == len(scores)

        return cls([GenerationOutput(text, score) for text, score in zip(texts, scores)])

    def update(self, other: Self):
        self.outputs.extend(other.outputs)

    def to_json(self) -> str:
        return json.dumps([output.to_json() for output in self.outputs])

    @classmethod
    def from_json(cls, json_obj: List[Dict[str, Any]]) -> Optional[Self]:
        return cls([GenerationOutput.from_json(obj) for obj in json_obj])

    def __repr__(self) -> str:
        return repr(self.outputs)


ModelOutput = ServerResponse


class ModelWrapper:
    MAX_BATCH_SIZE: int = 16

    model_name: str
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    device: torch.device
    generation_args: Dict[str, Any] = {
        "return_dict_in_generate": True,
        "output_scores": True,
        "renormalize_logits": True,
    }

    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def to(self, device: torch.device) -> Self:
        self.model = self.model.to(device)
        self.device = device
        return self

    def info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "settings": {"max_n": self.MAX_BATCH_SIZE}}

    def __call__(self, model_input: ModelInput) -> ModelOutput:
        output = ModelOutput.new([], [])
        for i in range(math.ceil(len(model_input.prompts) / self.MAX_BATCH_SIZE)):
            prompts = model_input.prompts[self.MAX_BATCH_SIZE * i : self.MAX_BATCH_SIZE * (i + 1)]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(
                **inputs, generation_config=model_input.generation_config, **self.generation_args
            )
            output.update(
                ModelOutput.new(
                    texts=self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True),
                    scores=list(torch.stack(outputs.scores, dim=1).cpu().numpy()),
                )
            )
        return output
