from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalityCsvSchemaProfile:
    modality: str
    accepted_identifier_columns: tuple[str, ...]
    min_columns: int = 2
    min_data_rows: int = 1

    def normalized_identifier_columns(self) -> tuple[str, ...]:
        return tuple(column.strip().lower() for column in self.accepted_identifier_columns)


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
    modality: ModalityCsvSchemaProfile(
        modality=modality,
        accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
    )
    for modality in (
        "T1w",
        "T1wHierarchical",
        "DTI",
        "rsfMRI",
        "T2Flair",
        "NM2DMT",
        "perf",
        "pet3d",
    )
}

_DEFAULT_PROFILE = ModalityCsvSchemaProfile(
    modality="default",
    accepted_identifier_columns=_GENERIC_IDENTIFIER_COLUMNS,
)


def get_schema_profile(modality: str) -> ModalityCsvSchemaProfile:
    return _PROFILES.get(str(modality), _DEFAULT_PROFILE)
