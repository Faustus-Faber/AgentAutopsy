"""Nanbeige4.2-3B integration for AgentDojo via vLLM.

NanbeigeLocalLLM is identical to GemmaLocalLLM/QwenLocalLLM in every respect
that affects the benchmark — same shared system prompt, same custom tool-call
parser, same sampling defaults (temperature=0.0, top_p=0.9), same retry budget,
same per-call metrics schema, same InfraFailure classification, same reading
of `message.reasoning` for thinking traces.

The model uses vLLM's native reasoning parser (`--reasoning-parser nanbeige`)
which populates `message.reasoning` automatically via the Qwen3 parser adapter
(same `<think>`/`</think>` token format).
"""

from gemma_llm import GemmaLocalLLM


class NanbeigeLocalLLM(GemmaLocalLLM):
    """Nanbeige4.2-3B wrapper. Inherits all behavior from GemmaLocalLLM unchanged.

    Fairness: by inheriting verbatim, this guarantees Nanbeige, Qwen, and Gemma
    traverse identical code paths for prompting, parsing, retrying, error
    classification, metrics tracking, and thinking extraction. The only
    differences are the model weights and vLLM's choice of reasoning parser.
    """
    pass
