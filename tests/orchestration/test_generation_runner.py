from pathlib import Path
from unittest.mock import patch

import pytest

from src.orchestration.generation_runner import generate_one_run


def test_generate_one_run_limits_loaded_samples() -> None:
    config = {"dataset": {}, "model": {"device_map": {"": 0}}}
    samples = [{"id": index} for index in range(3)]

    with (
        patch("src.orchestration.generation_runner.load_config", return_value=config),
        patch(
            "src.orchestration.generation_runner.load_run_samples",
            return_value=samples,
        ),
        patch("src.models.generation_pipeline.generate_run") as generate_run,
    ):
        generate_one_run(Path("run"), limit=1)

    generate_run.assert_called_once()
    assert generate_run.call_args.args[2] == samples[:1]


@pytest.mark.parametrize("mode", ["chat", "plain"])
@pytest.mark.parametrize("explicit_stop", [False, True])
def test_generation_keeps_chat_tokens_and_model_stop_tokens(
    tmp_path, mode, explicit_stop
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from src.models.generation_pipeline import generate_task

    backend = Tokenizer(
        WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A", special_tokens=[("<s>", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="</s>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=2,
        )
    ).eval()
    model.generation_config.eos_token_id = 2 if explicit_stop else [2, 3]
    model.generation_config.sequence_bias = {(3,): 100.0}
    config = {
        "model": {"name": "test"},
        "generation": {"max_new_tokens": 3},
        "capture": {"enabled": False},
        "prompt": {"mode": mode},
    }
    if explicit_stop:
        config["generation"]["eos_token_id"] = [2, 3]
    with patch("src.models.generation_pipeline.save_generation_output"):
        output = generate_task(
            run_path=tmp_path,
            config=config,
            model=model,
            tokenizer=tokenizer,
            sample={"id": "test", "question": "hello"},
            sample_index=0,
            sample_iter=0,
            progress=None,
            progress_label="",
        )
    assert output.input_ids == [1, 3]
    # Token 3 ends the model's turn, but is not tokenizer.eos_token_id (2).
    assert output.generated_token_ids == [3]
