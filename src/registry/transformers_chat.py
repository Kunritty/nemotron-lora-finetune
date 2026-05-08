"""
Transformers-backed chat model with a LangChain-shaped ``invoke(messages)`` API.

Use with :class:`chain.reasoning_chain.ReasoningChain` as a drop-in alternative to
other LangChain chat models.
"""

from __future__ import annotations

from typing import Any

import torch
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["TransformersChatLLM"]


def _text_content(msg: BaseMessage) -> str:
    raw = msg.content
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(raw)


def _messages_to_hf_chat(messages: list[BaseMessage]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            role = "user"
        out.append({"role": role, "content": _text_content(m)})
    return out


def _fallback_user_prompt(chat: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for turn in chat:
        blocks.append(f"{turn['role'].upper()}:\n{turn['content']}\n")
    blocks.append("ASSISTANT:\n")
    return "\n".join(blocks)


class TransformersChatLLM:
    """
    Loads a causal LM + tokenizer and implements ``invoke(list[BaseMessage]) -> AIMessage``.

    Tokenization uses ``tokenizer.apply_chat_template`` when available; otherwise a plain
    text fallback so older checkpoints still run.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.95,
        device_map: str | dict[str, Any] | None = "auto",
        torch_dtype: Any = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p

        dtype: Any
        if torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = torch_dtype

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="eager",
        )
        self._model.eval()

    @property
    def pretrained_model_name_or_path(self) -> str:
        return self._pretrained_model_name_or_path

    def invoke(
        self,
        messages: list[BaseMessage],
        config: dict[str, Any] | None = None,
    ) -> AIMessage:
        del config
        chat = _messages_to_hf_chat(messages)

        if getattr(self._tokenizer, "chat_template", None):
            input_ids = self._tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            prompt = _fallback_user_prompt(chat)
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("tokenization must return a torch.Tensor for input_ids")

        dev = next(self._model.parameters()).device
        input_ids = input_ids.to(dev)
        input_len = int(input_ids.shape[-1])

        do_sample = self._temperature is not None and self._temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self._temperature)
            gen_kwargs["top_p"] = float(self._top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self._model.generate(input_ids, **gen_kwargs)

        new_tokens = outputs[0, input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return AIMessage(content=text, additional_kwargs={})

