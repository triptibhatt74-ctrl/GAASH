"""
The only place that talks to OpenAI. Uses AsyncOpenAI + Structured Outputs
so the FastAPI event loop is never blocked and the response is guaranteed
to conform to NLPAnalysis (or the call raises, which /chat turns into a 502
rather than passing malformed data downstream).
"""
from openai import AsyncOpenAI, APIError, APITimeoutError

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TIMEOUT_SECONDS
from schemas import NLPAnalysis
from system_prompt import SYSTEM_PROMPT

_client = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT_SECONDS)


class LLMServiceError(Exception):
    """Raised for any failure that should surface as HTTP 502 to the
    client — timeout, API error, refusal, or a response that doesn't
    parse into NLPAnalysis."""


async def run_nlp_analysis(context_messages: list[dict]) -> NLPAnalysis:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *context_messages]
    try:
        completion = await _client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=messages,
            response_format=NLPAnalysis,
        )
    except (APIError, APITimeoutError) as exc:
        raise LLMServiceError(f"OpenAI request failed: {exc}") from exc

    choice = completion.choices[0]
    if choice.message.refusal:
        raise LLMServiceError(f"Model refused: {choice.message.refusal}")

    parsed = choice.message.parsed
    if parsed is None:
        raise LLMServiceError("Structured output failed to parse.")

    return parsed
