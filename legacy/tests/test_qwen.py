import asyncio
import json

import pytest
from pydantic import ValidationError
from resumeanalysis_legacy.inference.backends import MatchingProfile, create_predictor
from resumeanalysis_legacy.integration.applypilot import LocalInferenceClient
from resumeanalysis_legacy.qwen.config import QwenConfig
from resumeanalysis_legacy.qwen.predictor import InvalidModelOutput, QwenPredictor, skill_evidence
from resumeanalysis_legacy.qwen.prompts import Decision, build_messages

from resumeanalysis.cover_letters.qwen_runtime import TransformersGenerator
from resumeanalysis.jsonio import loads
from resumeanalysis.schemas import Profile, Vacancy


class FakeGenerator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, schema):
        self.calls.append((messages, schema))
        return next(self.responses)


def response(**kwargs):
    return json.dumps(
        {
            "relevant": True,
            "relevance": 0.8,
            "matched_skills": ["python"],
            "unknown_skills": ["docker"],
            "contradicted_skills": [],
            "evidence": [{"path": "/profile/skills/0", "quote": "Python"}],
            **kwargs,
        }
    )


def test_prompt_ru_en_and_explicit_profile_extension():
    p = MatchingProfile(
        skills=["Python"], preferred_roles=["ML Engineer"], resume_text="Опыт разработки"
    )
    v = Vacancy(description="Required: Python, Docker")
    messages = build_messages(p, v, {"matched_skills": ["python"]}, 0.5)
    data = loads(messages[1]["content"])
    assert data["input"]["profile"]["preferred_roles"] == ["ML Engineer"]
    assert "Опыт разработки" in messages[1]["content"]
    assert "untrusted" in messages[0]["content"] and "not a probability" in messages[0]["content"]
    with pytest.raises(ValidationError):
        Profile(preferred_roles=["ML Engineer"])
    assert "preferred_roles" not in Profile().model_dump()


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "```json\n{}\n```",
        "{}",
        response(relevance=1.1),
        response(relevance="0.8"),
        response(relevant="true"),
        response(matched_skills=["python", "rust"]),
        response(unknown_skills=[]),
        response(evidence=[{"path": "/profile/skills/0", "quote": "Rust"}]),
        response(evidence=[{"path": "/system/prompt", "quote": "Python"}]),
        response(relevant=False),
        response(extra="unsupported"),
        response().replace('"relevance": 0.8', '"relevance": NaN'),
        response().replace('"relevance": 0.8', '"relevance": 0.8, "relevance": 0.9'),
        response() + " trailing",
    ],
)
def test_invalid_model_output_fails_without_fallback_or_input_leak(raw):
    fake = FakeGenerator([raw])
    model = QwenPredictor(QwenConfig(retries=0), fake)
    with pytest.raises(InvalidModelOutput, match="validation") as error:
        model.predict(
            {"skills": ["Python"], "resume_text": "private phrase"},
            {"required_skills": ["Python", "Docker"]},
        )
    assert "private phrase" not in str(error.value)
    assert len(fake.calls) == 1


def test_mocked_predict_retries_and_grounded_unknown_semantics():
    fake = FakeGenerator(["invalid", response()])
    model = create_predictor("qwen", config=QwenConfig(retries=1), generator=fake)
    result = model.predict(
        {"skills": ["Python"], "preferred_roles": ["Engineer"]},
        {"required_skills": ["Python", "Docker"]},
    )
    assert result.relevant and result.relevance == 0.8
    assert result.unknown_skills == result.missing_skills == ["docker"]
    assert result.contradicted_skills == []
    assert "not proof" in result.explanation and "uncalibrated" in result.explanation
    assert result.summary is result.resume_id is result.cover_letter is None
    assert len(fake.calls) == 2


def test_empty_candidate_cannot_borrow_vacancy_qualifications():
    good = response(
        relevant=False, relevance=0.0, matched_skills=[], unknown_skills=["python"], evidence=[]
    )
    model = QwenPredictor(QwenConfig(retries=0), FakeGenerator([good]))
    result = model.predict({}, {"title": "Python engineer", "required_skills": ["Python"]})
    assert not result.relevant and result.unknown_skills == ["python"]
    bad = response(matched_skills=[], unknown_skills=["python"], evidence=[])
    with pytest.raises(InvalidModelOutput):
        QwenPredictor(QwenConfig(retries=0), FakeGenerator([bad])).predict(
            {}, {"required_skills": ["Python"]}
        )


def test_short_russian_resume_conflicts_and_preference_gating():
    detail = skill_evidence(
        Profile(resume_text="Знаю Python. Не знаю Docker.", skills=["Docker"]),
        Vacancy(required_skills=["Python", "Docker"]),
    )
    assert detail.matched == {"python"} and detail.unknown == {"docker"}
    result = QwenPredictor(generator=FakeGenerator([response()])).predict(
        {"skills": ["Python"], "remote_only": True}, {"required_skills": ["Python", "Docker"]}
    )
    assert not result.relevant and result.preference_unknowns
    assert result.relevance == 0.8


def test_explicit_denial_is_not_unknown():
    raw = response(unknown_skills=[], contradicted_skills=["docker"])
    result = QwenPredictor(generator=FakeGenerator([raw])).predict(
        {"skills": ["Python"], "denied_skills": ["Docker"]},
        {"required_skills": ["Python", "Docker"]},
    )
    assert result.contradicted_skills == ["docker"] and result.unknown_skills == []


def test_missing_weights_and_invalid_runtime_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        TransformersGenerator(QwenConfig(model_path=str(tmp_path / "missing")))
    with pytest.raises(ValidationError):
        QwenConfig(device="cpu")
    with pytest.raises(ValueError, match="Backend"):
        create_predictor("unknown")


def test_applypilot_adapter_uses_injected_local_profile():
    class Provider:
        def get_profile(self):
            return Profile(skills=["Python"])

    client = LocalInferenceClient(QwenPredictor(generator=FakeGenerator([response()])), Provider())
    result = asyncio.run(client.process(Vacancy(required_skills=["Python", "Docker"])))
    assert result.matched_skills == ["python"]
    assert Decision.model_validate(loads(response())).relevant


def test_actual_local_tiny_qwen_runtime_freezing_context_and_offline_loading(tmp_path):
    # Random tiny weights exercise Transformers mechanics, never model quality.
    from resumeanalysis_legacy.data.download import sha256_file
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    vocab = {s: i for i, s in enumerate(["<unk>", "<eos>", *sorted(ByteLevel.alphabet())])}
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteDecoder()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", eos_token="<eos>")
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}assistant"
    )
    fast.save_pretrained(tmp_path)
    Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
            eos_token_id=1,
        )
    ).save_pretrained(tmp_path)
    files = {p.name: sha256_file(p) for p in tmp_path.iterdir() if p.is_file()}
    (tmp_path / "resumeanalysis_qwen.json").write_text(
        json.dumps(
            {
                "model_id": "test-only-random-qwen",
                "revision": "test-fixture",
                "license": "apache-2.0",
                "sha256": files,
            }
        )
    )
    config = QwenConfig(
        model_id="test-only-random-qwen",
        revision="test-fixture",
        model_path=str(tmp_path),
        device="cpu",
        quantization="none",
        dtype="float32",
        max_input_tokens=64,
        max_new_tokens=128,
        structured_generation=False,
    )
    runtime = TransformersGenerator(config)
    assert all(not p.requires_grad for p in runtime.model.parameters())
    assert isinstance(runtime.generate([{"role": "user", "content": "hello"}], {}), str)
    # Exercise the actual optional JSON grammar integration, not pretrained quality.
    pytest.importorskip("lmformatenforcer")
    config.structured_generation = True
    runtime = TransformersGenerator(config)
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    constrained = loads(runtime.generate([{"role": "user", "content": "hello"}], schema))
    assert set(constrained) == {"ok"} and type(constrained["ok"]) is bool
    with pytest.raises(ValueError, match="no silent truncation"):
        runtime.generate([{"role": "user", "content": "hello " * 100}], {})
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        TransformersGenerator(config)
