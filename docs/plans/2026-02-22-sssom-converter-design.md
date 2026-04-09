# SSSOM to KB Converter - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `sssom_converter` module that converts SSSOM mapping TSV files into boomer `KB` objects, with configurable probability assignment.

**Architecture:** A `SSSOMConverter` class with a pydantic `SSSOMConverterConfig` that controls predicate-to-fact-type mapping, confidence-to-probability transforms, and source-based rule overrides. Pure logic module with no CLI dependency. SSSOM TSV parsed directly (no sssom-py dependency) since the format is simple: YAML metadata block + TSV rows.

**Tech Stack:** pydantic, PyYAML, standard library csv/io

---

### Task 1: SSSOM TSV parser

Parse SSSOM TSV files into structured data. SSSOM format: optional YAML metadata block (lines starting with `#`), then a TSV header row, then data rows.

**Files:**
- Create: `src/boomer/sssom_converter.py`
- Create: `tests/input/test_mappings.sssom.tsv`
- Create: `tests/test_sssom_converter.py`

**Step 1: Create test fixture**

Create a minimal SSSOM TSV file at `tests/input/test_mappings.sssom.tsv`:

```tsv
#curie_map:
#  MONDO: http://purl.obolibrary.org/obo/MONDO_
#  ORDO: http://www.orpha.net/ORDO/Orphanet_
#  OMIM: https://omim.org/entry/
#  skos: http://www.w3.org/2004/02/skos/core#
#  semapv: https://w3id.org/semapv/vocab/
#mapping_set_id: https://example.org/test_mappings
#mapping_set_description: Test mappings for boomer SSSOM converter
subject_id	subject_label	predicate_id	object_id	object_label	mapping_justification	confidence
ORDO:123	Alpha disease	skos:exactMatch	MONDO:0001234	Alpha disorder	semapv:LexicalMatching	0.95
ORDO:456	Beta disease	skos:broadMatch	MONDO:0001234	Alpha disorder	semapv:LexicalMatching	0.8
ORDO:789	Gamma disease	skos:exactMatch	MONDO:0005678	Gamma disorder	semapv:ManualMappingCuration	0.99
OMIM:100100	Delta syndrome	skos:exactMatch	MONDO:0001234	Alpha disorder	semapv:LexicalMatching	0.7
ORDO:123	Alpha disease	skos:narrowMatch	MONDO:0005678	Gamma disorder	semapv:SemanticSimilarityThresholdMatching	0.4
```

**Step 2: Write failing test for parser**

```python
import pytest
from pathlib import Path
from boomer.sssom_converter import parse_sssom_tsv

SSSOM_FILE = Path(__file__).parent / "input" / "test_mappings.sssom.tsv"


def test_parse_sssom_tsv():
    metadata, rows = parse_sssom_tsv(SSSOM_FILE)
    assert metadata["mapping_set_id"] == "https://example.org/test_mappings"
    assert len(rows) == 5
    row0 = rows[0]
    assert row0["subject_id"] == "ORDO:123"
    assert row0["predicate_id"] == "skos:exactMatch"
    assert row0["object_id"] == "MONDO:0001234"
    assert row0["confidence"] == "0.95"
    assert row0["subject_label"] == "Alpha disease"
```

**Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_sssom_converter.py::test_parse_sssom_tsv -v`
Expected: FAIL (module doesn't exist yet)

**Step 4: Implement parser**

In `src/boomer/sssom_converter.py`:

```python
"""Convert SSSOM mapping files to boomer KBs."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import yaml


def parse_sssom_tsv(path: str | Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """
    Parse a SSSOM TSV file into metadata and row dicts.

    SSSOM TSV format: lines starting with '#' contain YAML metadata,
    followed by a TSV header row and data rows.

    >>> metadata, rows = parse_sssom_tsv("tests/input/test_mappings.sssom.tsv")
    >>> metadata["mapping_set_id"]
    'https://example.org/test_mappings'
    >>> len(rows)
    5
    >>> rows[0]["subject_id"]
    'ORDO:123'

    Args:
        path: Path to the SSSOM TSV file.

    Returns:
        Tuple of (metadata_dict, list_of_row_dicts).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    metadata_lines = []
    tsv_lines = []
    for line in text.splitlines():
        if line.startswith("#"):
            # Strip the leading '#' (and optional space) for YAML parsing
            metadata_lines.append(line[1:])
        else:
            tsv_lines.append(line)

    metadata: dict[str, Any] = {}
    if metadata_lines:
        yaml_text = "\n".join(metadata_lines)
        parsed = yaml.safe_load(yaml_text)
        if isinstance(parsed, dict):
            metadata = parsed

    rows: list[dict[str, str]] = []
    if tsv_lines:
        reader = csv.DictReader(io.StringIO("\n".join(tsv_lines)), delimiter="\t")
        for row in reader:
            rows.append(dict(row))

    return metadata, rows
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_sssom_converter.py::test_parse_sssom_tsv -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/boomer/sssom_converter.py tests/test_sssom_converter.py tests/input/test_mappings.sssom.tsv
git commit -m "feat: add SSSOM TSV parser"
```

---

### Task 2: Confidence transforms

Implement configurable functions that map SSSOM confidence values to boomer prior probabilities.

**Files:**
- Modify: `src/boomer/sssom_converter.py`
- Modify: `tests/test_sssom_converter.py`

**Step 1: Write failing tests for transforms**

```python
from boomer.sssom_converter import (
    identity_transform,
    floor_ceil_transform,
    rescale_transform,
)


@pytest.mark.parametrize("confidence, expected", [
    (0.0, 0.0),
    (0.5, 0.5),
    (1.0, 1.0),
])
def test_identity_transform(confidence, expected):
    assert identity_transform(confidence) == pytest.approx(expected)


@pytest.mark.parametrize("confidence, floor, ceil, expected", [
    (0.0, 0.05, 0.95, 0.05),
    (1.0, 0.05, 0.95, 0.95),
    (0.5, 0.05, 0.95, 0.5),
    (0.01, 0.1, 0.9, 0.1),
])
def test_floor_ceil_transform(confidence, floor, ceil, expected):
    fn = floor_ceil_transform(floor, ceil)
    assert fn(confidence) == pytest.approx(expected)


@pytest.mark.parametrize("confidence, low, high, expected", [
    (0.0, 0.3, 0.95, 0.3),
    (1.0, 0.3, 0.95, 0.95),
    (0.5, 0.3, 0.95, 0.625),
])
def test_rescale_transform(confidence, low, high, expected):
    fn = rescale_transform(low, high)
    assert fn(confidence) == pytest.approx(expected)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sssom_converter.py -k "transform" -v`
Expected: FAIL

**Step 3: Implement transforms**

Add to `src/boomer/sssom_converter.py`:

```python
from typing import Callable

ConfidenceTransformFn = Callable[[float], float]


def identity_transform(confidence: float) -> float:
    """
    Pass confidence through unchanged.

    >>> identity_transform(0.8)
    0.8
    """
    return confidence


def floor_ceil_transform(floor: float = 0.05, ceil: float = 0.95) -> ConfidenceTransformFn:
    """
    Clamp confidence to [floor, ceil] range.

    >>> fn = floor_ceil_transform(0.05, 0.95)
    >>> fn(0.01)
    0.05
    >>> fn(0.99)
    0.95
    >>> fn(0.5)
    0.5
    """
    def transform(confidence: float) -> float:
        return max(floor, min(ceil, confidence))
    return transform


def rescale_transform(low: float = 0.3, high: float = 0.95) -> ConfidenceTransformFn:
    """
    Linearly rescale confidence from [0, 1] to [low, high].

    A confidence of 0 maps to `low` (uncertain but nonzero),
    a confidence of 1 maps to `high` (strong but not certain).

    >>> fn = rescale_transform(0.3, 0.95)
    >>> fn(0.0)
    0.3
    >>> fn(1.0)
    0.95
    """
    def transform(confidence: float) -> float:
        return low + confidence * (high - low)
    return transform
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_sssom_converter.py -k "transform" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/boomer/sssom_converter.py tests/test_sssom_converter.py
git commit -m "feat: add confidence-to-probability transform functions"
```

---

### Task 3: Config model and mapping rules

Define `MappingRule` and `SSSOMConverterConfig` pydantic models.

**Files:**
- Modify: `src/boomer/sssom_converter.py`
- Modify: `tests/test_sssom_converter.py`

**Step 1: Write failing test for config**

```python
from boomer.sssom_converter import SSSOMConverterConfig, MappingRule


def test_default_config():
    config = SSSOMConverterConfig()
    assert "skos:exactMatch" in config.predicate_defaults
    assert config.predicate_defaults["skos:exactMatch"] == 0.9
    assert config.auto_disjoint_groups is True
    assert config.min_probability == 0.01


def test_config_with_rules():
    config = SSSOMConverterConfig(
        rules=[
            MappingRule(subject_source="OMIM", probability=0.95),
            MappingRule(
                mapping_justification="semapv:LexicalMatching",
                confidence_transform="rescale",
                transform_params={"low": 0.2, "high": 0.7},
            ),
            MappingRule(predicate_id="skos:relatedMatch", skip=True),
        ]
    )
    assert len(config.rules) == 3
    assert config.rules[0].probability == 0.95
    assert config.rules[2].skip is True


def test_config_from_yaml():
    yaml_str = """
predicate_defaults:
  skos:exactMatch: 0.85
rules:
  - subject_source: OMIM
    probability: 0.95
"""
    config = SSSOMConverterConfig.model_validate(
        yaml.safe_load(yaml_str)
    )
    assert config.predicate_defaults["skos:exactMatch"] == 0.85
    assert config.rules[0].subject_source == "OMIM"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_sssom_converter.py -k "config" -v`
Expected: FAIL

**Step 3: Implement config models**

Add to `src/boomer/sssom_converter.py`:

```python
from pydantic import BaseModel, Field

# Default predicate -> fact type mapping
PREDICATE_FACT_MAP: dict[str, str] = {
    "skos:exactMatch": "EquivalentTo",
    "skos:closeMatch": "EquivalentTo",
    "skos:broadMatch": "ProperSubClassOf",   # subject < object
    "skos:narrowMatch": "ProperSubClassOf",  # object < subject (reversed)
    "owl:equivalentClass": "EquivalentTo",
    "rdfs:subClassOf": "ProperSubClassOf",
}

# Default priors when no confidence is given
DEFAULT_PREDICATE_PROBS: dict[str, float] = {
    "skos:exactMatch": 0.9,
    "skos:closeMatch": 0.7,
    "skos:broadMatch": 0.7,
    "skos:narrowMatch": 0.7,
    "owl:equivalentClass": 0.9,
    "rdfs:subClassOf": 0.8,
}


class MappingRule(BaseModel):
    """
    A rule for overriding probability assignment based on mapping properties.

    Rules are evaluated in order; first matching rule wins.

    >>> rule = MappingRule(subject_source="OMIM", probability=0.95)
    >>> rule.matches({"subject_id": "OMIM:100100", "predicate_id": "skos:exactMatch"})
    True
    >>> rule.matches({"subject_id": "ORDO:123", "predicate_id": "skos:exactMatch"})
    False
    """

    subject_source: str | None = None
    object_source: str | None = None
    predicate_id: str | None = None
    mapping_justification: str | None = None

    probability: float | None = None
    confidence_transform: str | None = None
    transform_params: dict[str, float] | None = None
    skip: bool = False

    def matches(self, row: dict[str, str]) -> bool:
        """Check if this rule matches a SSSOM row."""
        if self.subject_source and not row.get("subject_id", "").startswith(self.subject_source + ":"):
            return False
        if self.object_source and not row.get("object_id", "").startswith(self.object_source + ":"):
            return False
        if self.predicate_id and row.get("predicate_id") != self.predicate_id:
            return False
        if self.mapping_justification and row.get("mapping_justification") != self.mapping_justification:
            return False
        return True


class SSSOMConverterConfig(BaseModel):
    """
    Configuration for SSSOM to KB conversion.

    >>> config = SSSOMConverterConfig()
    >>> config.predicate_defaults["skos:exactMatch"]
    0.9
    >>> config.auto_disjoint_groups
    True
    """

    predicate_defaults: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_PREDICATE_PROBS))
    default_confidence_transform: str = "identity"
    default_transform_params: dict[str, float] | None = None
    rules: list[MappingRule] = Field(default_factory=list)
    auto_disjoint_groups: bool = True
    min_probability: float = 0.01
    subject_prefixes: list[str] | None = None
    object_prefixes: list[str] | None = None
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_sssom_converter.py -k "config" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/boomer/sssom_converter.py tests/test_sssom_converter.py
git commit -m "feat: add SSSOMConverterConfig and MappingRule models"
```

---

### Task 4: Core conversion logic - `sssom_to_kb()`

The main conversion function that ties parser, transforms, rules, and fact generation together.

**Files:**
- Modify: `src/boomer/sssom_converter.py`
- Modify: `tests/test_sssom_converter.py`

**Step 1: Write failing tests**

```python
from boomer.sssom_converter import sssom_to_kb, SSSOMConverterConfig, MappingRule
from boomer.model import EquivalentTo, ProperSubClassOf, MemberOfDisjointGroup


def test_sssom_to_kb_basic():
    """Basic conversion with default config."""
    kb = sssom_to_kb(SSSOM_FILE)
    assert kb.name is not None

    # Should have pfacts for each mapping row
    assert len(kb.pfacts) == 5

    # Check first mapping: ORDO:123 exactMatch MONDO:0001234 @ 0.95
    equiv_pfacts = [p for p in kb.pfacts if isinstance(p.fact, EquivalentTo)
                    and p.fact.sub == "ORDO:123" and p.fact.equivalent == "MONDO:0001234"]
    assert len(equiv_pfacts) == 1
    assert equiv_pfacts[0].prob == pytest.approx(0.95)

    # Check broadMatch -> ProperSubClassOf(subject, object)
    sub_pfacts = [p for p in kb.pfacts if isinstance(p.fact, ProperSubClassOf)
                  and p.fact.sub == "ORDO:456" and p.fact.sup == "MONDO:0001234"]
    assert len(sub_pfacts) == 1
    assert sub_pfacts[0].prob == pytest.approx(0.8)

    # Check narrowMatch -> ProperSubClassOf(object, subject)
    narrow_pfacts = [p for p in kb.pfacts if isinstance(p.fact, ProperSubClassOf)
                     and p.fact.sub == "MONDO:0005678" and p.fact.sup == "ORDO:123"]
    assert len(narrow_pfacts) == 1
    assert narrow_pfacts[0].prob == pytest.approx(0.4)

    # Labels should be extracted
    assert kb.labels["ORDO:123"] == "Alpha disease"
    assert kb.labels["MONDO:0001234"] == "Alpha disorder"


def test_sssom_to_kb_disjoint_groups():
    """Auto-generated disjoint groups from prefixes."""
    kb = sssom_to_kb(SSSOM_FILE)
    disjoint_facts = [f for f in kb.facts if isinstance(f, MemberOfDisjointGroup)]
    # Should have one per unique entity, grouped by prefix
    ordo_members = [f for f in disjoint_facts if f.group == "ORDO"]
    mondo_members = [f for f in disjoint_facts if f.group == "MONDO"]
    omim_members = [f for f in disjoint_facts if f.group == "OMIM"]
    assert len(ordo_members) == 3   # ORDO:123, ORDO:456, ORDO:789
    assert len(mondo_members) == 2  # MONDO:0001234, MONDO:0005678
    assert len(omim_members) == 1   # OMIM:100100


def test_sssom_to_kb_with_rules():
    """Rules override probability assignment."""
    config = SSSOMConverterConfig(
        rules=[
            MappingRule(subject_source="OMIM", probability=0.95),
        ]
    )
    kb = sssom_to_kb(SSSOM_FILE, config=config)
    omim_pfacts = [p for p in kb.pfacts if isinstance(p.fact, EquivalentTo)
                   and p.fact.sub == "OMIM:100100"]
    assert len(omim_pfacts) == 1
    assert omim_pfacts[0].prob == pytest.approx(0.95)


def test_sssom_to_kb_with_skip_rule():
    """Rules can skip mappings."""
    config = SSSOMConverterConfig(
        rules=[
            MappingRule(predicate_id="skos:narrowMatch", skip=True),
        ]
    )
    kb = sssom_to_kb(SSSOM_FILE, config=config)
    # Should have 4 pfacts (5 minus the skipped narrowMatch)
    assert len(kb.pfacts) == 4


def test_sssom_to_kb_with_transform():
    """Confidence transform applied to probabilities."""
    config = SSSOMConverterConfig(
        default_confidence_transform="rescale",
        default_transform_params={"low": 0.3, "high": 0.95},
    )
    kb = sssom_to_kb(SSSOM_FILE, config=config)
    # ORDO:123 exactMatch MONDO:0001234 @ confidence 0.95
    # rescale(0.95) = 0.3 + 0.95 * 0.65 = 0.9175
    equiv_pfacts = [p for p in kb.pfacts if isinstance(p.fact, EquivalentTo)
                    and p.fact.sub == "ORDO:123" and p.fact.equivalent == "MONDO:0001234"]
    assert equiv_pfacts[0].prob == pytest.approx(0.9175)


def test_sssom_to_kb_no_confidence():
    """Falls back to predicate defaults when no confidence column."""
    # Create a SSSOM without confidence column - test with rows directly
    from boomer.sssom_converter import sssom_mappings_to_pfacts
    rows = [
        {"subject_id": "A:1", "predicate_id": "skos:exactMatch", "object_id": "B:1"},
    ]
    config = SSSOMConverterConfig()
    pfacts = sssom_mappings_to_pfacts(rows, config=config)
    assert len(pfacts) == 1
    assert pfacts[0].prob == pytest.approx(0.9)  # default for exactMatch


def test_sssom_to_kb_prefix_filter():
    """Subject/object prefix filters."""
    config = SSSOMConverterConfig(
        subject_prefixes=["ORDO"],
    )
    kb = sssom_to_kb(SSSOM_FILE, config=config)
    # Should only have ORDO subjects (3 rows: ORDO:123 exact, ORDO:456 broad, ORDO:789 exact, ORDO:123 narrow)
    assert len(kb.pfacts) == 4  # excludes OMIM:100100 row


def test_sssom_to_kb_min_probability():
    """Pfacts below min_probability are dropped."""
    config = SSSOMConverterConfig(min_probability=0.5)
    kb = sssom_to_kb(SSSOM_FILE, config=config)
    # The narrowMatch row has confidence 0.4 which is below threshold
    assert all(p.prob >= 0.5 for p in kb.pfacts)
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_sssom_converter.py -k "sssom_to_kb" -v`
Expected: FAIL

**Step 3: Implement conversion logic**

Add to `src/boomer/sssom_converter.py`:

```python
from boomer.model import (
    KB,
    PFact,
    EquivalentTo,
    ProperSubClassOf,
    MemberOfDisjointGroup,
)
from boomer.io import id_prefix


NAMED_TRANSFORMS: dict[str, Callable[..., ConfidenceTransformFn]] = {
    "identity": lambda **_: identity_transform,
    "floor_ceil": lambda **params: floor_ceil_transform(**params),
    "rescale": lambda **params: rescale_transform(**params),
}


def _resolve_transform(name: str, params: dict[str, float] | None = None) -> ConfidenceTransformFn:
    """Resolve a named transform to a callable."""
    factory = NAMED_TRANSFORMS.get(name)
    if factory is None:
        raise ValueError(f"Unknown confidence transform: {name!r}. Available: {list(NAMED_TRANSFORMS)}")
    return factory(**(params or {}))


def _make_fact(predicate_id: str, subject_id: str, object_id: str):
    """
    Create a boomer Fact from a SSSOM predicate.

    >>> _make_fact("skos:exactMatch", "A:1", "B:1")
    EquivalentTo(fact_type='EquivalentTo', sub='A:1', equivalent='B:1')
    >>> _make_fact("skos:broadMatch", "A:1", "B:1")
    ProperSubClassOf(fact_type='ProperSubClassOf', sub='A:1', sup='B:1')
    >>> _make_fact("skos:narrowMatch", "A:1", "B:1")
    ProperSubClassOf(fact_type='ProperSubClassOf', sub='B:1', sup='A:1')
    """
    fact_type = PREDICATE_FACT_MAP.get(predicate_id)
    if fact_type is None:
        return None
    if fact_type == "EquivalentTo":
        return EquivalentTo(sub=subject_id, equivalent=object_id)
    elif fact_type == "ProperSubClassOf":
        if predicate_id == "skos:narrowMatch":
            return ProperSubClassOf(sub=object_id, sup=subject_id)
        else:
            return ProperSubClassOf(sub=subject_id, sup=object_id)
    return None


def sssom_mappings_to_pfacts(
    rows: list[dict[str, str]],
    config: SSSOMConverterConfig | None = None,
) -> list[PFact]:
    """
    Convert parsed SSSOM rows to a list of PFacts.

    >>> rows = [{"subject_id": "A:1", "predicate_id": "skos:exactMatch", "object_id": "B:1", "confidence": "0.9"}]
    >>> pfacts = sssom_mappings_to_pfacts(rows)
    >>> pfacts[0].prob
    0.9
    >>> pfacts[0].fact.fact_type
    'EquivalentTo'
    """
    if config is None:
        config = SSSOMConverterConfig()

    default_transform = _resolve_transform(
        config.default_confidence_transform,
        config.default_transform_params,
    )

    pfacts: list[PFact] = []
    for row in rows:
        subject_id = row.get("subject_id", "")
        object_id = row.get("object_id", "")
        predicate_id = row.get("predicate_id", "")

        # Apply prefix filters
        if config.subject_prefixes:
            if not any(subject_id.startswith(p + ":") for p in config.subject_prefixes):
                continue
        if config.object_prefixes:
            if not any(object_id.startswith(p + ":") for p in config.object_prefixes):
                continue

        # Find first matching rule
        matched_rule: MappingRule | None = None
        for rule in config.rules:
            if rule.matches(row):
                matched_rule = rule
                break

        # Skip if rule says so
        if matched_rule and matched_rule.skip:
            continue

        # Create fact from predicate
        fact = _make_fact(predicate_id, subject_id, object_id)
        if fact is None:
            continue

        # Determine probability
        prob: float
        if matched_rule and matched_rule.probability is not None:
            prob = matched_rule.probability
        elif "confidence" in row and row["confidence"]:
            confidence = float(row["confidence"])
            if matched_rule and matched_rule.confidence_transform:
                transform = _resolve_transform(
                    matched_rule.confidence_transform,
                    matched_rule.transform_params,
                )
                prob = transform(confidence)
            else:
                prob = default_transform(confidence)
        else:
            prob = config.predicate_defaults.get(predicate_id, 0.5)

        if prob < config.min_probability:
            continue

        pfacts.append(PFact(fact=fact, prob=prob))

    return pfacts


def sssom_to_kb(
    path: str | Path,
    config: SSSOMConverterConfig | None = None,
) -> KB:
    """
    Convert a SSSOM TSV file to a boomer KB.

    >>> kb = sssom_to_kb("tests/input/test_mappings.sssom.tsv")
    >>> len(kb.pfacts)
    5
    >>> kb.labels["ORDO:123"]
    'Alpha disease'

    Args:
        path: Path to the SSSOM TSV file.
        config: Optional converter configuration.

    Returns:
        A KB with pfacts from the mappings and MemberOfDisjointGroup facts.
    """
    if config is None:
        config = SSSOMConverterConfig()

    metadata, rows = parse_sssom_tsv(path)
    pfacts = sssom_mappings_to_pfacts(rows, config=config)

    # Extract labels
    labels: dict[str, str] = {}
    for row in rows:
        if row.get("subject_label"):
            labels[row["subject_id"]] = row["subject_label"]
        if row.get("object_label"):
            labels[row["object_id"]] = row["object_label"]

    # Auto-generate disjoint groups from prefixes
    facts = []
    if config.auto_disjoint_groups:
        seen_ids: set[str] = set()
        for row in rows:
            for id_key in ("subject_id", "object_id"):
                entity_id = row.get(id_key, "")
                if entity_id and entity_id not in seen_ids:
                    seen_ids.add(entity_id)
                    try:
                        prefix = id_prefix(entity_id)
                        facts.append(MemberOfDisjointGroup(sub=entity_id, group=prefix))
                    except ValueError:
                        pass

    # KB metadata from SSSOM metadata block
    name = metadata.get("mapping_set_id", Path(path).stem)
    description = metadata.get("mapping_set_description")

    return KB(
        facts=facts,
        pfacts=pfacts,
        labels=labels,
        name=name,
        description=description,
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_sssom_converter.py -v`
Expected: PASS

**Step 5: Run doctests**

Run: `uv run pytest --doctest-modules src/boomer/sssom_converter.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/boomer/sssom_converter.py tests/test_sssom_converter.py
git commit -m "feat: add sssom_to_kb() converter with configurable transforms and rules"
```

---

### Task 5: Config file loading and rule-based transform test

Add ability to load config from a YAML file, and a test that exercises the full rule-based transform pipeline with a realistic config.

**Files:**
- Modify: `src/boomer/sssom_converter.py`
- Create: `tests/input/sssom_config.yaml`
- Modify: `tests/test_sssom_converter.py`

**Step 1: Create test config fixture**

Create `tests/input/sssom_config.yaml`:

```yaml
predicate_defaults:
  skos:exactMatch: 0.85
  skos:broadMatch: 0.6
  skos:narrowMatch: 0.6

default_confidence_transform: floor_ceil
default_transform_params:
  floor: 0.05
  ceil: 0.95

rules:
  - subject_source: OMIM
    probability: 0.95
  - mapping_justification: "semapv:LexicalMatching"
    confidence_transform: rescale
    transform_params:
      low: 0.2
      high: 0.7

auto_disjoint_groups: true
min_probability: 0.05
```

**Step 2: Write failing test**

```python
def test_config_from_file():
    config_path = Path(__file__).parent / "input" / "sssom_config.yaml"
    config = load_sssom_config(config_path)
    assert config.predicate_defaults["skos:exactMatch"] == 0.85
    assert len(config.rules) == 2


def test_sssom_to_kb_with_config_file():
    """Full pipeline: SSSOM file + config file -> KB."""
    config_path = Path(__file__).parent / "input" / "sssom_config.yaml"
    config = load_sssom_config(config_path)
    kb = sssom_to_kb(SSSOM_FILE, config=config)

    # OMIM row should use hard override probability 0.95
    omim_pfacts = [p for p in kb.pfacts if isinstance(p.fact, EquivalentTo)
                   and p.fact.sub == "OMIM:100100"]
    assert len(omim_pfacts) == 1
    assert omim_pfacts[0].prob == pytest.approx(0.95)

    # ORDO:123 exactMatch MONDO:0001234 has confidence=0.95 and justification=LexicalMatching
    # LexicalMatching rule applies: rescale(0.95, low=0.2, high=0.7) = 0.2 + 0.95 * 0.5 = 0.675
    ordo_exact = [p for p in kb.pfacts if isinstance(p.fact, EquivalentTo)
                  and p.fact.sub == "ORDO:123" and p.fact.equivalent == "MONDO:0001234"]
    assert len(ordo_exact) == 1
    assert ordo_exact[0].prob == pytest.approx(0.675)

    # ORDO:789 exactMatch with ManualMappingCuration, confidence=0.99
    # No rule matches, falls back to default transform: floor_ceil(0.99, 0.05, 0.95) = 0.95
    manual_exact = [p for p in kb.pfacts if isinstance(p.fact, EquivalentTo)
                    and p.fact.sub == "ORDO:789"]
    assert len(manual_exact) == 1
    assert manual_exact[0].prob == pytest.approx(0.95)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_sssom_converter.py -k "config_file" -v`
Expected: FAIL

**Step 4: Implement config loader**

Add to `src/boomer/sssom_converter.py`:

```python
def load_sssom_config(path: str | Path) -> SSSOMConverterConfig:
    """
    Load a SSSOMConverterConfig from a YAML file.

    >>> config = load_sssom_config("tests/input/sssom_config.yaml")
    >>> config.predicate_defaults["skos:exactMatch"]
    0.85
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return SSSOMConverterConfig.model_validate(data)
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_sssom_converter.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/boomer/sssom_converter.py tests/test_sssom_converter.py tests/input/sssom_config.yaml
git commit -m "feat: add SSSOM config file loading and rule-based transform pipeline"
```

---

### Task 6: Lint and doctest pass

**Step 1: Run linter**

Run: `uv run ruff check src/boomer/sssom_converter.py`
Fix any issues.

**Step 2: Run doctests**

Run: `uv run pytest --doctest-modules src/boomer/sssom_converter.py -v`
Fix any issues.

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Ensure nothing is broken.

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore: lint and doctest fixes for sssom_converter"
```

---

### Task 7: Update skill documentation

Update the pyboomer skill to document SSSOM support.

**Files:**
- Modify: `~/.claude/skills/pyboomer/SKILL.md`

**Step 1: Add SSSOM section to skill**

Add a section after "Ptable Format" covering:
- Direct SSSOM solving: `pyboomer solve mappings.sssom.tsv -t 60`
- Using a config file: `pyboomer solve mappings.sssom.tsv --sssom-config config.yaml -t 60`
- Programmatic usage: `sssom_to_kb(path, config=config)`
- Config file format with example

**Note:** CLI integration itself is a future task. For now document the Python API usage that agents can use via `uv run python -c "..."` or in scripts.

**Step 2: Commit**

```bash
git add ~/.claude/skills/pyboomer/SKILL.md
git commit -m "docs: add SSSOM support to pyboomer skill"
```
