from __future__ import annotations

from contextlib import nullcontext
from typing import Any


class HFGenerator:
    def __init__(self, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_name = config.get("torch_dtype", "auto")
        dtype = getattr(torch, dtype_name) if dtype_name != "auto" else "auto"
        kwargs = {
            "torch_dtype": dtype,
            "device_map": config.get("device_map", "auto"),
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        }
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=kwargs["trust_remote_code"])
        self.model = AutoModelForCausalLM.from_pretrained(config["model_name"], **kwargs).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, config: dict[str, Any], seed: int, temperature: float) -> dict[str, Any]:
        torch = self.torch
        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": int(config.get("max_new_tokens", 256)),
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = output.sequences[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return {
            "text": text,
            "token_ids": generated_ids.detach().cpu().tolist(),
            "logprobs": self._chosen_logprobs(output.scores, generated_ids),
            "activations": self._activations(inputs["input_ids"][0], generated_ids, config),
        }

    def _chosen_logprobs(self, scores: tuple[Any, ...], token_ids: Any) -> list[float]:
        values = []
        for score, token_id in zip(scores, token_ids):
            logprobs = self.torch.log_softmax(score[0], dim=-1)
            values.append(float(logprobs[token_id].detach().cpu()))
        return values

    def _activations(self, prompt_ids: Any, generated_ids: Any, config: dict[str, Any]) -> dict[str, Any]:
        layers = [int(layer) for layer in config.get("layers", [])]
        if not layers:
            return {}
        input_ids = self.torch.cat([prompt_ids, generated_ids]).unsqueeze(0).to(self.model.device)
        ctx = self.torch.autocast("cuda", enabled=input_ids.device.type == "cuda") if hasattr(self.torch, "autocast") else nullcontext()
        with self.torch.no_grad(), ctx:
            output = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        start = int(prompt_ids.shape[0])
        activations = {}
        for layer in layers:
            hidden = output.hidden_states[layer][0, start:, :]
            activations[str(layer)] = hidden.float().detach().cpu().numpy()
        return activations
