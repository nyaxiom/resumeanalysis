"""Loss-aware JSONL adapters. Source schemas and canonical schemas are separate.

All diagnostics use row identifiers and field paths, never source values/text.
Raw retention is an explicit local policy, independent of conversion permissiveness.
"""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from resumeanalysis.jsonio import loads
from resumeanalysis.schemas import InputModel, Profile, Vacancy
from resumeanalysis_legacy.data.dataset import SPLITS, PairRecord


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def fields(value, path="") -> dict:
    """Enumerate nested fields, including empty containers and object-array fields.

    Primitive arrays are atomic fields; object arrays use wildcard schema paths.
    Values are not placed in reports.
    """
    if isinstance(value, dict) and value:
        result = {}
        for key, child in value.items():
            for name, observations in fields(child, path + "/" + escape(key)).items():
                result.setdefault(name, []).extend(observations)
        return result
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        result = {path: [value]}
        for item in value:
            for name, observations in fields(item, path + "/*").items():
                result.setdefault(name, []).extend(observations)
        return result
    return {path: [value]}


def get_path(value, path):
    for part in path[1:].split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def put_path(value, path, child):
    parts = [v.replace("~1", "/").replace("~0", "~") for v in path[1:].split("/")]
    for part in parts[:-1]:
        value = value.setdefault(part, {})
    if parts[-1] in value:
        raise ValueError("Two source mappings target one canonical field")
    value[parts[-1]] = child


class Rule(InputModel):
    source: str = Field(pattern=r"^/[^*]+$")
    target: str | None = None
    disposition: Literal["direct", "transformed", "auxiliary", "excluded"] = "direct"
    transform: Literal["identity", "string", "float", "integer", "label"] = "identity"
    reason: str | None = None
    source_types: list[Literal["str", "int", "float", "bool", "list", "NoneType"]] | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.disposition in {"direct", "transformed"}:
            if not self.target or not self.target.startswith("/"):
                raise ValueError("Mapped rules require a canonical JSON Pointer target")
        elif not self.reason or self.target is not None or self.transform != "identity":
            raise ValueError(
                "Auxiliary/excluded fields require a reason and no transformation/target"
            )
        if self.transform != "identity" and self.disposition != "transformed":
            raise ValueError("Type/label transformations must be explicitly documented")
        if self.transform != "identity" and not self.source_types:
            raise ValueError("Transformed fields must declare accepted source_types")
        return self


class LabelMapping(InputModel):
    source: str | int | bool
    target: Literal[0, 1]


class Exclusion(InputModel):
    field: str
    equals: str | int | bool | None
    reason: str = Field(min_length=1)


class AdapterSpec(InputModel):
    name: str = "mapped-jsonl"
    rules: list[Rule] = Field(default_factory=list)
    labels: list[LabelMapping] = Field(default_factory=list)
    exclusions: list[Exclusion] = Field(default_factory=list)
    schema_version_field: str | None = None
    expected_schema_version: str | int | None = None

    @model_validator(mode="after")
    def unique(self):
        if len({r.source for r in self.rules}) != len(self.rules):
            raise ValueError("Duplicate source mapping")
        targets = [r.target for r in self.rules if r.target]
        if len(set(targets)) != len(targets):
            raise ValueError("Duplicate target mapping")
        # Parent/child rules would overlap and obscure information loss.
        for left in self.rules:
            for right in self.rules:
                if left != right and right.source.startswith(left.source + "/"):
                    raise ValueError("Overlapping source rules are unsupported")
                if (
                    left.target
                    and right.target
                    and left != right
                    and right.target.startswith(left.target + "/")
                ):
                    raise ValueError("Overlapping target rules are unsupported")
        keys = [json.dumps(m.source) for m in self.labels]
        if len(set(keys)) != len(keys):
            raise ValueError("Ambiguous label mapping")
        return self


def canonical_spec() -> AdapterSpec:
    rules = [
        Rule(source="/" + name, target="/" + name)
        for name in PairRecord.model_fields
        if name not in {"profile", "vacancy"}
    ]
    for prefix, model in [("profile", Profile), ("vacancy", Vacancy)]:
        rules.extend(
            Rule(source=f"/{prefix}/{name}", target=f"/{prefix}/{name}")
            for name in model.model_fields
        )
    return AdapterSpec(name="canonical", rules=rules)


@dataclass
class SourceRow:
    source_id: str
    file: str
    line: int
    sha256: str
    file_sha256: str
    split: str | None
    raw: bytes
    value: object = None
    error: str | None = None


def source_file_policy(path) -> dict:
    path = Path(path)
    return {
        "format": "UTF-8 JSONL; top-level *.jsonl when input is a directory",
        "excluded_entries": [
            {"name": entry.name, "reason": "Outside the declared top-level JSONL input format"}
            for entry in sorted(path.iterdir())
            if not entry.is_file() or entry.suffix != ".jsonl"
        ]
        if path.is_dir()
        else [],
    }


class SourceNumberError(ValueError):
    pass


def source_float(token):
    number = float(token)
    if not math.isfinite(number) or Decimal(token) != Decimal(str(number)):
        raise SourceNumberError("JSON number would lose decimal precision")
    return number


def source_rows(path: str | Path, dataset: str, revision: str) -> list[SourceRow]:
    path = Path(path)
    files = sorted(p for p in path.glob("*.jsonl") if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError("No JSONL source files found")
    rows = []
    for file in files:
        contents = file.read_bytes()
        file_hash = hashlib.sha256(contents).hexdigest()
        for number, raw in enumerate(contents.splitlines(keepends=True), 1):
            checksum = hashlib.sha256(raw).hexdigest()
            identifier = digest([dataset, revision, file.name, file_hash, number, checksum])
            row = SourceRow(
                identifier,
                file.name,
                number,
                checksum,
                file_hash,
                file.stem if file.stem in SPLITS else None,
                raw,
            )
            try:
                row.value = loads(raw.decode("utf-8"), parse_float=source_float)
                if not isinstance(row.value, dict):
                    raise ValueError("Expected JSON object")
            except SourceNumberError:
                row.error = "numeric_precision_or_range_loss"
            except (ValueError, UnicodeError):
                row.error = "invalid_json_object"  # Blank lines are accounted for too.
            rows.append(row)
    if not rows:
        raise ValueError("Source contains zero records")
    return rows


def _rule_for(path: str, spec: AdapterSpec) -> Rule | None:
    if spec.name == "canonical" and path in {"/profile", "/vacancy"}:
        return Rule(source=path, target=path)
    return next(
        (
            r
            for r in spec.rules
            if path == r.source
            or (r.disposition in {"auxiliary", "excluded"} and path.startswith(r.source + "/"))
        ),
        None,
    )


def inspect_rows(rows: list[SourceRow], spec: AdapterSpec) -> dict:
    coverage = {}
    for row in rows:
        if row.error:
            continue
        for path, values in fields(row.value).items():
            rule = _rule_for(path, spec)
            entry = coverage.setdefault(
                path,
                {
                    "disposition": rule.disposition if rule else "unsupported",
                    "target": rule.target if rule else None,
                    "reason": rule.reason if rule else None,
                    "records_present": 0,
                    "null_values": 0,
                    "missing_or_empty_values": 0,
                    "types": Counter(),
                    "source_ids": [],
                },
            )
            entry["records_present"] += 1
            entry["null_values"] += sum(v is None for v in values)
            entry["missing_or_empty_values"] += sum(
                v is None or v == "" or v == [] or v == {} for v in values
            )
            entry["types"].update(type(v).__name__ for v in values)
            if len(entry["source_ids"]) < 3:
                entry["source_ids"].append(row.source_id)
    valid = sum(r.error is None for r in rows)
    for path in {r.source for r in spec.rules} | set(coverage):
        if path not in coverage:
            rule = _rule_for(path, spec)
            coverage[path] = {
                "disposition": rule.disposition,
                "target": rule.target,
                "reason": rule.reason,
                "records_present": 0,
                "null_values": 0,
                "missing_or_empty_values": 0,
                "types": {},
                "source_ids": [],
            }
        coverage[path]["records_missing"] = valid - coverage[path]["records_present"]
        coverage[path]["missing_rate"] = (
            (valid - coverage[path]["records_present"]) / valid if valid else None
        )
    observed = [v for v in coverage.values() if v["records_present"]]
    return {
        "total_input_records": len(rows),
        "invalid_source_records": len(rows) - valid,
        "source_error_counts": dict(Counter(r.error for r in rows if r.error)),
        "schema_coverage": coverage,
        "field_coverage_fraction": sum(v["disposition"] != "unsupported" for v in observed)
        / len(observed)
        if observed
        else 0,
        "warning": "Schema coverage measures disposition, not information available to the model.",
    }


def _transform(value, rule, spec):
    if rule.transform == "identity":
        return value
    if rule.transform == "label":
        matches = [
            m.target for m in spec.labels if type(m.source) is type(value) and m.source == value
        ]
        if len(matches) != 1:
            raise ValueError("unmapped_label")
        return matches[0]
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("unsupported_type_conversion")
    if rule.transform == "string":
        return str(value)
    if rule.transform == "integer":
        number = Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError("invalid_numeric_conversion")
        return int(number)
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("invalid_numeric_conversion")
    return number


def convert_row(row: SourceRow, spec: AdapterSpec, permissive: bool) -> tuple:
    if row.error:
        return None, "rejected", [{"code": row.error}]
    observed = fields(row.value)
    issues = []
    unknown = [p for p in observed if not _rule_for(p, spec)]
    issues.extend({"code": "unmapped_field", "field": p} for p in unknown)
    if spec.schema_version_field:
        try:
            actual = get_path(row.value, spec.schema_version_field)
        except (KeyError, TypeError):
            actual = None
        if (
            type(actual) is not type(spec.expected_schema_version)
            or actual != spec.expected_schema_version
        ):
            return None, "rejected", [*issues, {"code": "schema_version_change"}]
    if unknown and not permissive:
        return None, "rejected", issues
    for policy in spec.exclusions:
        try:
            actual = get_path(row.value, policy.field)
        except (KeyError, TypeError):
            continue
        if type(actual) is type(policy.equals) and actual == policy.equals:
            return (
                None,
                "excluded",
                [*issues, {"code": "policy_exclusion", "reason": policy.reason}],
            )
    converted = {}
    if spec.name == "canonical":
        for name in ("profile", "vacancy"):
            if isinstance(row.value.get(name), dict):
                converted[name] = {}
    for rule in spec.rules:
        try:
            value = get_path(row.value, rule.source)
        except (KeyError, TypeError):
            continue  # Missing required canonical values fail validation below.
        if rule.disposition in {"excluded", "auxiliary"}:
            issues.append(
                {"code": rule.disposition + "_field", "field": rule.source, "reason": rule.reason}
            )
            continue
        if rule.source_types and type(value).__name__ not in rule.source_types:
            return None, "rejected", [*issues, {"code": "unexpected_type", "field": rule.source}]
        # Mapping whole dictionaries/arrays of objects could hide nested fields.
        if isinstance(value, dict) or (
            isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value)
        ):
            return (
                None,
                "rejected",
                [*issues, {"code": "unsupported_nested_mapping", "field": rule.source}],
            )
        try:
            mapped = _transform(value, rule, spec)
            put_path(converted, rule.target, mapped)
        except (ValueError, TypeError, ArithmeticError):
            return None, "rejected", [*issues, {"code": "conversion_error", "field": rule.source}]
        if rule.source != rule.target:
            issues.append({"code": "renamed_field", "field": rule.source, "target": rule.target})
        if rule.transform != "identity":
            issues.append(
                {
                    "code": "transformed_field",
                    "field": rule.source,
                    "transform": rule.transform,
                    "source_type": type(value).__name__,
                    "target_type": type(mapped).__name__,
                }
            )
    try:
        if type(converted.get("label")) is not int:
            raise ValueError("invalid_label")
        result = PairRecord.model_validate(converted, strict=True)
    except ValidationError as exc:
        return (
            None,
            "rejected",
            [
                *issues,
                *[
                    {
                        "code": "canonical_validation",
                        "field": "/" + "/".join(map(str, e["loc"])),
                        "error_type": e["type"],
                    }
                    for e in exc.errors(include_input=False)
                ],
            ],
        )
    except ValueError:
        return None, "rejected", [*issues, {"code": "invalid_label"}]
    normalized = result.model_dump()
    for rule in spec.rules:
        if not rule.target:
            continue
        try:
            before, after = get_path(converted, rule.target), get_path(normalized, rule.target)
        except (KeyError, TypeError):
            continue
        if type(before) is not type(after) or before != after:
            issues.append(
                {
                    "code": "canonical_type_conversion",
                    "field": rule.source,
                    "target": rule.target,
                    "source_type": type(before).__name__,
                    "target_type": type(after).__name__,
                }
            )
    loss = any(
        i["code"]
        in {"unmapped_field", "excluded_field", "transformed_field", "canonical_type_conversion"}
        for i in issues
    )
    return result, "converted_with_loss" if loss else "converted", issues
