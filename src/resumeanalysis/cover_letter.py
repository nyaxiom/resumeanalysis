"""One lazy Transformers/Qwen backend adapted from the existing local runtime."""

import re
import threading

from resumeanalysis.config import Settings
from resumeanalysis.prompts import build_messages


class ContextTooLong(ValueError):
    pass


class InvalidLetter(RuntimeError):
    pass


class Qwen:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = self.tokenizer = None
        self.lock = threading.Lock()

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self.settings
        device = s.qwen_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and s.qwen_dtype == "float16":
            raise ValueError("CPU generation requires float32 or bfloat16")
        source = str(s.qwen_local_path or s.qwen_model)
        options = dict(
            revision=s.qwen_revision,
            local_files_only=bool(s.qwen_local_path),
            trust_remote_code=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(source, **options)
        model = AutoModelForCausalLM.from_pretrained(
            source,
            use_safetensors=True,
            torch_dtype=getattr(torch, s.qwen_dtype),
            **options,
        )
        model.eval().requires_grad_(False).to(device)
        self.tokenizer, self.model = tokenizer, model

    def generate(self, messages):
        import torch

        with self.lock:
            self._load()
            s = self.settings
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                return_token_type_ids=False,
            )
            size = tokens.input_ids.shape[1]
            context = getattr(self.model.config, "max_position_embeddings", s.qwen_max_input_tokens)
            if size > s.qwen_max_input_tokens or size + s.qwen_max_new_tokens > context:
                raise ContextTooLong(
                    "Input exceeds the Qwen token budget; shorten the supplied text"
                )
            tokens = tokens.to(self.model.device)
            with torch.inference_mode():
                result = self.model.generate(
                    **tokens,
                    max_new_tokens=s.qwen_max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generated = result[0, size:]
            eos = self.model.generation_config.eos_token_id
            eos = [eos] if isinstance(eos, int) else eos or [self.tokenizer.eos_token_id]
            if len(generated) >= s.qwen_max_new_tokens and int(generated[-1]) not in eos:
                raise InvalidLetter("Generation reached its token limit before finishing")
            return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class CoverLetters:
    def __init__(self, settings: Settings, backend=None):
        self.settings = settings
        self.backend = backend if backend is not None else Qwen(settings)

    def generate(self, request):
        output = self.backend.generate(build_messages(request, self.settings.letter_max_words))
        if not isinstance(output, str) or not output.strip():
            raise InvalidLetter("The model returned an empty letter")
        text = output.strip()
        if len(text.split()) > self.settings.letter_max_words:
            raise InvalidLetter("The letter exceeds the word limit")
        if text.startswith(("{", "[", "```")) or re.search(r"</?think>|\[[^\]\n]+\]", text, re.I):
            raise InvalidLetter("The model returned markup, reasoning, or placeholders")

        # Bounded guard, not a semantic verifier. It cannot establish whether
        # arbitrary paraphrases are entailed by the resume.
        def numbers(value):
            return set(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?(?:%)?", value))

        if numbers(text) - numbers(request.resume_text):
            raise InvalidLetter("The letter contains an unsupported numeric claim")
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|https?://", text):
            raise InvalidLetter("The letter contains contact details or links")
        return text
