"""LLM Agent — wraps model inference for SENTINEL.

Provides a clean interface that:
  1. Converts the Gymnasium obs dict → text prompt (prompt_builder)
  2. Runs model.generate() with sampling params
  3. Parses the output → valid Action dict (action_parser)
  4. Falls back to UCB1+Bayesian math when GPU/model unavailable

Usage:
    agent = LLMAgent(model, tokenizer, agent_role="holmes")
    action = agent.act(obs, step=5)

The class is designed to be used inside _execute_episode() in pipeline.py.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sentinel.training.action_parser import parse_llm_action
from sentinel.training.prompt_builder import build_messages, build_prompt

if TYPE_CHECKING:
    pass  # model/tokenizer type stubs not available without unsloth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generation config defaults
# ---------------------------------------------------------------------------

_DEFAULT_GEN_KWARGS: dict[str, Any] = {
    "max_new_tokens": 256,    # JSON action is short; cap to avoid rambling
    "temperature":    0.3,    # lower = more deterministic JSON output
    "top_p":          0.9,
    "do_sample":      True,
    "pad_token_id":   0,      # overridden per tokenizer in __init__
}


class LLMAgent:
    """Wraps a HuggingFace / Unsloth LLM for SENTINEL action generation.

    Args:
        model: A loaded (optionally LoRA-adapted) HuggingFace model.
        tokenizer: Matching tokenizer (must support apply_chat_template).
        agent_role: "holmes" | "forge" | "argus" | "oracle" | "hermes".
        device: "cuda" | "cpu" — where to run inference.
        use_chat_template: If True, use tokenizer.apply_chat_template().
                           If False, use raw prompt string (for older models).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        agent_role: str = "holmes",
        device: str = "cuda",
        use_chat_template: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.agent_role = agent_role
        self.device = device
        self.use_chat_template = use_chat_template
        self._step_counter = 0

        # Patch pad_token_id for generation
        gen_kwargs = dict(_DEFAULT_GEN_KWARGS)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

        # Stop generation at the closing fence so the model doesn't ramble past the JSON
        stop_strings = ["```", "}\n```", "}\n\n"]
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            gen_kwargs["stop_strings"] = stop_strings
            gen_kwargs["tokenizer"] = tokenizer

        self._gen_kwargs = gen_kwargs

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def act(self, obs: dict[str, Any], step: int = 0) -> dict[str, Any]:
        """Generate one action from the current observation.

        Args:
            obs: Raw Gymnasium observation dict.
            step: Current episode step (added to prompt for context).

        Returns:
            Valid action dict: {agent, category, name, params, _llm_completion}.
            The ``_llm_completion`` key holds the raw model output for GRPO
            sample collection; it is stripped by _execute_episode before
            passing the action to env.step().
        """
        self._step_counter = step
        try:
            raw_output = self._generate(obs, step)
            action = parse_llm_action(raw_output, fallback_agent=self.agent_role)
            # Attach raw completion for GRPO training data collection
            action["_llm_completion"] = raw_output
            logger.debug(
                "[LLMAgent/%s] step=%d → %s.%s",
                self.agent_role, step, action["agent"], action["name"],
            )
            return action
        except Exception as exc:
            logger.warning(
                "[LLMAgent/%s] Inference error at step %d: %s — using fallback.",
                self.agent_role, step, exc,
            )
            return self._fallback_action()

    def reset(self) -> None:
        """Reset per-episode state."""
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate(self, obs: dict[str, Any], step: int) -> str:
        """Run one forward pass through the model and return raw text."""
        import torch  # lazy import — only needed when GPU available

        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = build_messages(obs, self.agent_role, step)
            # continue_final_message=True honours the assistant pre-fill
            # ("```json\n{") so the model completes JSON, not prose.
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            except TypeError:
                # Older transformers versions don't have continue_final_message
                input_text = self.tokenizer.apply_chat_template(
                    messages[:-1],          # drop pre-filled assistant turn
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            input_text = build_prompt(obs, self.agent_role, step)

        # Tokenise
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=3072,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **self._gen_kwargs,
            )

        # Decode only the NEW tokens (skip input)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Re-attach the pre-fill prefix so the parser sees a complete block
        # e.g. model generates: '"agent": "holmes"...}```'
        #      we return:       '```json\n{"agent": "holmes"...}```'
        if self.use_chat_template:
            raw = "```json\n{" + raw

        return raw


    def _fallback_action(self) -> dict[str, Any]:
        """Return a safe default action matching this agent's role."""
        if self.agent_role == "forge":
            return {
                "agent":    "forge",
                "category": "remediation",
                "name":     "ScaleService",
                "params":   {"service": "api-gateway", "replicas": 2},
            }
        return {
            "agent":    "holmes",
            "category": "investigative",
            "name":     "QueryLogs",
            "params":   {"service": "api-gateway", "time_range": [0, 300]},
        }


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_llm_agent(
    model: Any,
    tokenizer: Any,
    agent_role: str = "holmes",
) -> LLMAgent | None:
    """Create an LLMAgent, returning None if model/tokenizer are unavailable."""
    if model is None or tokenizer is None:
        return None
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return LLMAgent(model, tokenizer, agent_role=agent_role, device=device)


# ---------------------------------------------------------------------------
# Reward wrapper for GRPOTrainer
# ---------------------------------------------------------------------------

def make_grpo_reward_fn(env: Any) -> Any:
    """Return a GRPO-compatible reward function that uses SENTINEL's reward system.

    GRPOTrainer expects:
        reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]

    This wrapper:
      1. Parses each completion as an action
      2. Executes it in a copy of the current env state (via rollout)
      3. Returns the step reward as the GRPO signal
    """
    def _reward_fn(
        prompts: list[str],
        completions: list[str],
        agent_role: str = "holmes",
        obs: dict | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            try:
                action = parse_llm_action(completion, fallback_agent=agent_role)
                # Use the last known obs for step reward approximation
                if obs is not None and hasattr(env, "reward_function"):
                    rf = env.reward_function
                    inc = env._incident_state
                    step_r = rf.compute_step_reward(action, env.world_state, inc)
                    rewards.append(float(step_r))
                else:
                    rewards.append(0.0)
            except Exception as exc:
                logger.debug("GRPO reward fn error: %s", exc)
                rewards.append(-0.1)
        return rewards

    return _reward_fn
