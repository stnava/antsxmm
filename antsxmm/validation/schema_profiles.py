from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalityCsvSchemaProfile:
    modality: str
    accepted_identifier_columns: tuple[str, ...]
    min_columns: int = 2
    min_data_rows: int = 1
    strict_metric_tokens: tuple[str, ...] = ()
    strict_min_metric_matches: int = 1

    def normalized_identifier_columns(self) -> tuple[str, ...]:
        return tuple(column.strip().lower() for column in self.accepted_identifier_columns)

    def normalized_metric_tokens(self) -> tuple[str, ...]:
        return tuple(token.strip().lower() for token in self.strict_metric_tokens if token.strip())

    def strict_enabled(self) -> bool:
        return bool(self.normalized_metric_tokens())

    def match_metric_columns(self, columns: tuple[str, ...]) -> tuple[str, ...]:
        tokens = self.normalized_metric_tokens()
        if not tokens:
            return ()
        matches: list[str] = []
        seen: set[str] = set()
        for column in columns:
            normalized = str(column).strip().lower()
            if not normalized:
                continue
            if any(token in normalized for token in tokens) and normalized not in seen:
                matches.append(column)
                seen.add(normalized)
        return tuple(matches)


_GENERIC_IDENTIFIER_COLUMNS = (
    "bids_subject",
    "subject_id",
    "metric",
    "measure",
    "feature",
    "label",
    "id",
)


_PROFILES: dict[str, ModalityCsvSchemaProfile] = {
    "T1w": ModalityCsvSchemaProfile(
        modality="T1w",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("vol", "volume", "thick", "thickness", "area", "snr", "cnr"),
        strict_min_metric_matches=1,
    ),
    "T1wHierarchical": ModalityCsvSchemaProfile(
        modality="T1wHierarchical",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("vol", "volume", "thick", "thickness", "area", "snr", "cnr"),
        strict_min_metric_matches=1,
    ),
    "DTI": ModalityCsvSchemaProfile(
        modality="DTI",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("fa", "md", "rd", "ad", "adc"),
        strict_min_metric_matches=1,
    ),
    "rsfMRI": ModalityCsvSchemaProfile(
        modality="rsfMRI",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("alff", "falff", "reho", "tsnr", "corr", "connect"),
        strict_min_metric_matches=1,
    ),
    "T2Flair": ModalityCsvSchemaProfile(
        modality="T2Flair",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("wmh", "lesion", "flair", "volume"),
        strict_min_metric_matches=1,
    ),
    "NM2DMT": ModalityCsvSchemaProfile(
        modality="NM2DMT",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("nm", "uptake", "binding", "suv", "suvr"),
        strict_min_metric_matches=1,
    ),
    "perf": ModalityCsvSchemaProfile(
        modality="perf",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("cbf", "att", "bat", "transit", "perfusion"),
        strict_min_metric_matches=1,
    ),
    "pet3d": ModalityCsvSchemaProfile(
        modality="pet3d",
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
        strict_metric_tokens=("suv", "suvr", "uptake", "binding"),
        strict_min_metric_matches=1,
    ),
}

_DEFAULT_PROFILE = ModalityCsvSchemaProfile(
    modality="default",
    accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
)


def get_schema_profile(modality: str) -> ModalityCsvSchemaProfile:
    return _PROFILES.get(str(modality), _DEFAULT_PROFILE)
