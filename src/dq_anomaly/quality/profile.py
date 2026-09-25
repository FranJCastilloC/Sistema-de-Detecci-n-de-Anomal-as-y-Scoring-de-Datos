"""Expectation profiles: what a given table is supposed to look like.

Profiles are authored in YAML (readable and diffable by whoever owns the data)
and validated with pydantic so a malformed profile fails loudly at load time
rather than halfway through a batch.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from dq_anomaly.config import PATHS, load_yaml

DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "completeness": 0.25,
    "validity": 0.25,
    "consistency": 0.20,
    "uniqueness": 0.15,
    "plausibility": 0.15,
}

# Expressions come from profile files, which are trusted input. The allowlist
# keeps that trust cheap to verify rather than assumed.
ALLOWED_FUNCTIONS = {"abs", "round"}
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


class ColumnExpectation(BaseModel):
    """What a single column must satisfy."""

    name: str
    dtype: Literal["int", "float", "string", "datetime", "bool"] = "string"
    required: bool = True
    max_null_rate: float = 0.0
    min: float | None = None
    max: float | None = None
    min_date: str | None = None
    max_date: str | None = None
    allowed_values: list[str] | None = None
    regex: str | None = None
    unique: bool = False
    weight: float = Field(default=1.0, gt=0)
    description: str = ""


class CrossFieldRule(BaseModel):
    """A relationship that must hold between columns, or across tables."""

    id: str
    kind: Literal["expression", "foreign_key", "lookup_match"]
    description: str
    impact: float = Field(default=0.8, ge=0.0, le=1.0)
    # kind == "expression"
    expression: str | None = None
    # kind == "foreign_key" / "lookup_match"
    column: str | None = None
    ref_table: str | None = None
    ref_column: str | None = None
    ref_key: str | None = None
    local_key: str | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> "CrossFieldRule":
        if self.kind == "expression" and not self.expression:
            raise ValueError(f"rule {self.id}: 'expression' is required")
        if self.kind in {"foreign_key", "lookup_match"}:
            missing = [f for f in ("column", "ref_table", "ref_column") if not getattr(self, f)]
            if missing:
                raise ValueError(f"rule {self.id}: missing {missing}")
        if self.kind == "lookup_match" and not (self.local_key and self.ref_key):
            raise ValueError(f"rule {self.id}: lookup_match needs local_key and ref_key")
        return self


class TableExpectation(BaseModel):
    """The full expectation suite for one table."""

    name: str
    description: str = ""
    primary_key: list[str] = Field(default_factory=list)
    columns: list[ColumnExpectation] = Field(default_factory=list)
    cross_field_rules: list[CrossFieldRule] = Field(default_factory=list)
    outlier_columns: list[str] = Field(default_factory=list)
    outlier_cut: float = 4.0
    dimension_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_DIMENSION_WEIGHTS)
    )
    inferred: bool = False

    @field_validator("dimension_weights")
    @classmethod
    def _known_dimensions(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = set(value) - set(DEFAULT_DIMENSION_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown quality dimensions: {sorted(unknown)}")
        return value

    @property
    def column_map(self) -> dict[str, ColumnExpectation]:
        return {column.name: column for column in self.columns}

    def validate_expressions(self) -> None:
        """Reject any expression referencing names outside this table."""
        known = set(self.column_map) | ALLOWED_FUNCTIONS
        for rule in self.cross_field_rules:
            if rule.kind != "expression":
                continue
            check_expression(rule.expression or "", known, rule.id)


def check_expression(expression: str, allowed_names: set[str], rule_id: str = "") -> ast.Expression:
    """Parse an expression and reject anything outside the allowlist."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - surfaced to the profile author
        raise ValueError(f"rule {rule_id}: cannot parse expression: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"rule {rule_id}: disallowed syntax {type(node).__name__} in expression"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError(f"rule {rule_id}: only {sorted(ALLOWED_FUNCTIONS)} may be called")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"rule {rule_id}: unknown name '{node.id}' in expression")
    return tree


def load_profile(path: str | Path) -> TableExpectation:
    """Load and validate a profile from YAML."""
    raw: dict[str, Any] = load_yaml(path)
    profile = TableExpectation.model_validate(raw)
    profile.validate_expressions()
    return profile


def _infer_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    return "string"


def infer_profile(df: pd.DataFrame, name: str = "uploaded_batch") -> TableExpectation:
    """Derive a best-effort profile from the data itself.

    An inferred profile can only measure a batch against its own shape, so it
    detects internal inconsistency, never incorrectness. Callers surface that
    distinction via the ``inferred`` flag.
    """
    columns: list[ColumnExpectation] = []
    outlier_columns: list[str] = []
    for column in df.columns:
        series = df[column]
        dtype = _infer_dtype(series)
        null_rate = float(series.isna().mean())
        expectation = ColumnExpectation(
            name=str(column),
            dtype=dtype,
            required=null_rate < 0.5,
            max_null_rate=min(1.0, round(null_rate + 0.02, 4)),
            description="Inferred from the uploaded batch",
        )
        if dtype in {"int", "float"}:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric) > 20:
                low, high = numeric.quantile([0.001, 0.999])
                span = max(abs(high - low), 1e-9)
                expectation.min = float(low - 0.2 * span)
                expectation.max = float(high + 0.2 * span)
                outlier_columns.append(str(column))
        elif dtype == "string":
            distinct = series.dropna().unique()
            if 0 < len(distinct) <= 20:
                expectation.allowed_values = sorted(str(value) for value in distinct)
        columns.append(expectation)

    primary_key: list[str] = []
    for column in df.columns:
        series = df[column]
        if series.notna().all() and series.is_unique:
            primary_key = [str(column)]
            break

    return TableExpectation(
        name=name,
        description="Profile inferred from the uploaded batch",
        primary_key=primary_key,
        columns=columns,
        outlier_columns=outlier_columns,
        inferred=True,
    )


def detect_profile(
    df: pd.DataFrame, profiles: list[TableExpectation], min_overlap: float = 0.9
) -> TableExpectation | None:
    """Return the registered profile whose required columns the batch matches."""
    present = {str(column) for column in df.columns}
    best: tuple[float, TableExpectation] | None = None
    for profile in profiles:
        required = {column.name for column in profile.columns if column.required}
        if not required:
            continue
        overlap = len(required & present) / len(required)
        if overlap >= min_overlap and (best is None or overlap > best[0]):
            best = (overlap, profile)
    return best[1] if best else None


def load_bundled_profiles(config_dir: Path | None = None) -> list[TableExpectation]:
    """Load every ``profile_*.yaml`` shipped in the config directory."""
    config_dir = Path(config_dir or PATHS.config)
    return [load_profile(path) for path in sorted(config_dir.glob("profile_*.yaml"))]
