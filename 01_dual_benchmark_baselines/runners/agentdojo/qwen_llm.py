"""Qwen3.5-4B integration for AgentDojo via vLLM.

QwenLocalLLM is identical to GemmaLocalLLM in every respect that affects
the benchmark — same shared system prompt, same custom tool-call parser,
same sampling defaults (temperature=0.0, top_p=0.9), same retry budget,
same per-call metrics schema, same InfraFailure classification, same
reading of `message.reasoning` for thinking traces.

Both models use vLLM's native reasoning parser to separate thinking
content (Gemma: `--reasoning-parser gemma4`, Qwen: `--reasoning-parser qwen3`)
which populates `message.reasoning` automatically.

A separate class exists purely to make the model identity explicit in
pipeline introspection (e.g., `isinstance(llm_elem, QwenLocalLLM)` in
the runner) and to keep the per-model file structure parallel.
"""

from gemma_llm import GemmaLocalLLM


class QwenLocalLLM(GemmaLocalLLM):
    """Qwen3.5-4B wrapper. Inherits all behavior from GemmaLocalLLM unchanged.

    Fairness: by inheriting verbatim, this guarantees Qwen and Gemma traverse
    identical code paths for prompting, parsing, retrying, error classification,
    metrics tracking, and thinking extraction. The only differences are the
    model weights and vLLM's choice of reasoning parser.
    """
    pass
