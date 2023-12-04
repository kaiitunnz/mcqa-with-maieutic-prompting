import backoff
import google.generativeai as palm  # type: ignore
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from google.generativeai.types import safety_types  # type: ignore

MODEL = "models/text-bison-001"
MAX_RETRY_COUNT = 4
TEMPERATURE = 0.3
SAFETY_SETTINGS = (
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


def configure(api_key: str):
    palm.configure(api_key=api_key)


@backoff.on_exception(backoff.expo, ServiceUnavailable, max_tries=MAX_RETRY_COUNT)
@backoff.on_exception(backoff.expo, ResourceExhausted)
def generate_text(
    prompt: str, candidate_count: int = 1, temperature: float = TEMPERATURE
):
    return palm.generate_text(
        model=MODEL,
        prompt=prompt,
        candidate_count=candidate_count,
        temperature=temperature,
        top_p=1,
        max_output_tokens=64,
        safety_settings=SAFETY_SETTINGS,
    )
