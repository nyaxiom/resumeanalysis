"""Parser and extraction contracts using authored documents and mocked LLM outputs."""

import hashlib
import io
import zipfile
from types import SimpleNamespace

import pytest

from resumeanalysis.profiles.documents import DocumentError, LocalDocumentParser
from resumeanalysis.profiles.extraction import (
    ExtractedProfile,
    ExtractionError,
    ProfileExtractionConfig,
    StructuredProfileExtractor,
)
from resumeanalysis.recommendation.schema import (
    EvidenceReference,
    ProfileData,
    StructuredProfile,
    StructuredVacancy,
)


class Client:
    def __init__(self, value):
        self.value, self.calls = value, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def response():
    return ExtractedProfile(
        data=ProfileData(facts={"skills": ["Python"]}),
        evidence={"/facts/skills/0": [EvidenceReference(source_id="user-text", quote="Python")]},
    )


def test_local_extraction_preserves_citations_and_requires_review():
    client = Client(response())
    result = StructuredProfileExtractor(client=client).extract_text("alice", "I use Python.")
    assert result.review_status == "needs_review" and result.facts.skills == ["Python"]
    assert result.facts.experience_years is None
    assert result.evidence["/facts/skills/0"][0].quote == "Python"
    assert client.calls[0]["response_model"] is ExtractedProfile
    assert client.calls[0]["max_retries"] == 1


@pytest.mark.parametrize("problem", ["missing", "quote", "source", "offset", "backend"])
def test_unverifiable_or_failed_extraction_is_rejected_without_private_error_details(problem):
    output = response()
    citation = output.evidence["/facts/skills/0"][0]
    if problem == "missing":
        output.evidence.clear()
    elif problem == "quote":
        citation.quote = "private unsupported skill"
    elif problem == "source":
        citation.source_id = "missing-source"
    elif problem == "offset":
        citation.start, citation.end = 0, 6
    elif problem == "backend":
        output = RuntimeError("private candidate details from provider")
    with pytest.raises(ExtractionError) as error:
        StructuredProfileExtractor(client=Client(output)).extract_text("alice", "I use Python.")
    assert "private" not in str(error.value)


def test_structured_input_and_user_correction_need_no_llm_and_preserve_original_source():
    profile = StructuredProfileExtractor.from_structured(
        "alice", {"facts": {"skills": ["Python"], "experience_years": 2}}
    )
    original = profile.model_dump()
    corrected = StructuredProfileExtractor.correct(
        profile, {"/facts/experience_years": 3, "/preferences/remote": True}, confirm=True
    )
    assert corrected.version == 2 and corrected.review_status == "confirmed"
    assert corrected.facts.experience_years == 3 and corrected.preferences.remote is True
    assert corrected.sources[0] == profile.sources[0]
    assert corrected.sources[-1].kind == "user_edit"
    assert corrected.evidence["/facts/experience_years"][0].source_id == "user-edit-2"
    assert profile.model_dump() == original


def test_remote_url_requires_explicit_opt_in_and_source_budget_is_enforced_before_client():
    with pytest.raises(ValueError, match="allow_remote"):
        ProfileExtractionConfig(base_url="https://provider.example/v1")
    assert ProfileExtractionConfig(base_url="http://[::1]:11434/v1")
    assert ProfileExtractionConfig(base_url="https://provider.example/v1", allow_remote=True)
    client = Client(response())
    extractor = StructuredProfileExtractor(
        ProfileExtractionConfig(max_source_characters=100), client=client
    )
    with pytest.raises(ExtractionError, match="budget"):
        extractor.extract_text("alice", "a" * 101)
    assert not client.calls


def test_strict_nested_fields_do_not_coerce_missing_or_false_values():
    with pytest.raises(ValueError):
        StructuredProfile(user_id="alice", facts={"experience_years": "3"})
    with pytest.raises(ValueError):
        StructuredVacancy(vacancy_id="one", facts={"remote": "false"})
    with pytest.raises(ValueError):
        StructuredProfileExtractor.from_structured("alice", {"facts": {"invented_field": 2}})


def test_actual_instructor_library_validates_mocked_local_provider_response():
    instructor = pytest.importorskip("instructor")
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")
    calls = []

    def handle(request):
        calls.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "authored",
                "object": "chat.completion",
                "created": 0,
                "model": "local-test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": response().model_dump_json()},
                    }
                ],
            },
        )

    with httpx2.Client(
        transport=httpx2.MockTransport(handle), trust_env=False, follow_redirects=False
    ) as transport:
        sdk = openai.OpenAI(
            base_url="http://127.0.0.1:11434/v1", api_key="local", http_client=transport
        )
        client = instructor.from_openai(sdk, mode=instructor.Mode.JSON)
        result = StructuredProfileExtractor(client=client).extract_text("alice", "I use Python.")
    assert result.facts.skills == ["Python"]
    assert len(calls) == 1 and calls[0].url.host == "127.0.0.1"


def test_default_sdk_constructor_uses_fixed_local_url_and_ignores_proxy_environment(monkeypatch):
    pytest.importorskip("instructor")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://external.example/v1")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example")
    extractor = StructuredProfileExtractor()
    client = extractor._client()
    assert str(client.client.base_url).startswith("http://127.0.0.1:11434/")
    assert not extractor._transport.trust_env
    assert not extractor._transport.follow_redirects
    extractor.close()


def test_txt_parsing_keeps_original_bytes_digest_and_text():
    data = "Python\nЯ использую SQL.".encode()
    result = LocalDocumentParser().parse(data, "candidate.txt", source_id="doc-1")
    assert (
        result.text == data.decode() and result.original_sha256 == hashlib.sha256(data).hexdigest()
    )


def test_docx_parses_paragraphs_and_tables_without_extracting_archives_to_disk():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Python experience")
    document.add_table(rows=1, cols=2).rows[0].cells[0].text = "SQL"
    buffer = io.BytesIO()
    document.save(buffer)
    result = LocalDocumentParser().parse(buffer.getvalue(), "candidate.docx", source_id="docx")
    assert "Python experience" in result.text and "SQL" in result.text


def pdf_bytes(text=True, pages=1):
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(300, 300)
        if text:
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 20 200 Td (Python and SQL) Tj ET")
            page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_extracts_text_with_page_offsets_and_rejects_empty_scans():
    result = LocalDocumentParser().parse(pdf_bytes(), "resume.pdf", source_id="pdf")
    assert "Python and SQL" in result.text
    page = result.pages[0]
    assert page["page"] == 1 and result.text[page["start"] : page["end"]] == result.text
    with pytest.raises(DocumentError, match="OCR"):
        LocalDocumentParser().parse(pdf_bytes(False), "scan.pdf", source_id="scan")
    with pytest.raises(DocumentError, match="page limit"):
        LocalDocumentParser(max_pages=1).parse(pdf_bytes(pages=2), "two.pdf", source_id="two")


@pytest.mark.parametrize(
    "data,name",
    [
        (b"bad", "a.pdf"),
        (b"bad", "a.docx"),
        (b"\xff", "a.txt"),
        (b"hello", "a.exe"),
        (b"", "a.txt"),
    ],
)
def test_malformed_and_unsupported_documents_fail_cleanly(data, name):
    with pytest.raises(DocumentError):
        LocalDocumentParser().parse(data, name, source_id="bad")


def test_document_size_and_archive_expansion_limits():
    with pytest.raises(DocumentError, match="byte limit"):
        LocalDocumentParser(max_bytes=3).parse(b"long", "a.txt", source_id="big")
    with pytest.raises(DocumentError, match="character limit"):
        LocalDocumentParser(max_characters=3).parse(b"long", "a.txt", source_id="big")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "a" * 100_000)
    with pytest.raises(DocumentError, match="expansion"):
        LocalDocumentParser().parse(buffer.getvalue(), "bomb.docx", source_id="bomb")


def test_provider_error_logs_do_not_disclose_document_content(caplog):
    import logging

    class LoggingClient(Client):
        def create(self, **kwargs):
            logging.getLogger("instructor.v2.core.retry").error("private resume payload")
            raise RuntimeError("private resume payload")

    with pytest.raises(ExtractionError):
        StructuredProfileExtractor(client=LoggingClient(None)).extract_text(
            "alice", "private resume payload"
        )
    assert "private resume payload" not in caplog.text
