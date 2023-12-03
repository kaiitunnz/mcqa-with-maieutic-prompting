import json
import os
from typing import Any, Dict, List, Optional

import requests
from transformers import GenerationConfig

from .base import BatchRequest, GenerationOutput, ModelOutput, ServerConfig, SingleRequest


class ModelClient:
    url: str
    timeout: int

    def __init__(self, config_file: Optional[str] = None, timeout: Optional[int] = None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        server_config = ServerConfig.new(config_file)
        self.url = f"http://{server_config.address}:{server_config.port}"
        self.timeout = timeout

    def info(self) -> Optional[Dict[str, Any]]:
        info_url = f"{self.url}/info"
        response = requests.get(info_url, timeout=self.timeout)
        if response.status_code == 200:
            return json.loads(response.text)
        return None

    def generate(self, prompt: str, generation_config: GenerationConfig, n: int = 1) -> List[GenerationOutput]:
        request = SingleRequest(prompt, generation_config, n)
        generate_url = f"{self.url}{request.path}"
        json_data = json.dumps(request.to_dict())
        response = requests.post(
            generate_url, data=json_data, headers={"Content-Type": "application/json"}, timeout=self.timeout
        )

        if response.status_code != 200:
            raise ValueError("Failed to communicate with the server.")
        response_data = json.loads(response.text)
        return ModelOutput.from_json(response_data).outputs

    def batch_generate(
        self, prompts: List[str], generation_config: GenerationConfig, n: int = 1
    ) -> List[GenerationOutput]:
        request = BatchRequest(prompts, generation_config, n)
        generate_url = f"{self.url}{request.path}"
        json_data = json.dumps(request.to_dict())
        response = requests.post(
            generate_url, data=json_data, headers={"Content-Type": "application/json"}, timeout=self.timeout
        )

        if response.status_code != 200:
            raise ValueError("Failed to communicate with the server.")
        response_data = json.loads(response.text)
        return ModelOutput.from_json(response_data).outputs
