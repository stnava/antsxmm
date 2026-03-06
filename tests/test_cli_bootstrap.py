from __future__ import annotations

from pathlib import Path


def test_pyproject_console_script_points_to_cli_bootstrap():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert 'antsxmm = "antsxmm.cli:main"' in text


def test_cli_source_applies_environment_before_importing_pipeline():
    cli_path = Path(__file__).resolve().parents[1] / "antsxmm" / "cli.py"
    text = cli_path.read_text(encoding="utf-8")
    apply_idx = text.index("apply_default_environment()")
    import_idx = text.index("from .pipeline import entry_point")
    assert apply_idx < import_idx


def test_package_init_does_not_eagerly_import_pipeline_or_core():
    init_path = Path(__file__).resolve().parents[1] / "antsxmm" / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    pre_getattr, _, _ = text.partition("def __getattr__(name: str):")
    assert "from .pipeline import" not in pre_getattr
    assert "from .core import" not in pre_getattr
    assert "def __getattr__(name: str):" in text
