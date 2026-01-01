from pathlib import Path

try:
    import tomllib as tomli
except ModuleNotFoundError:  # pragma: no cover
    import tomli  # type: ignore


def test_pyproject_core_deps_do_not_include_training_stack():
    data = tomli.loads(Path("pyproject.toml").read_text())
    deps = [d.lower() for d in data["project"]["dependencies"]]

    assert not any(d.startswith("lightning") for d in deps)
    assert not any(d.startswith("tensorboard") for d in deps)
    assert not any(d.startswith("wandb") for d in deps)
    assert not any(d.startswith("jsonargparse") for d in deps)
    assert not any(d.startswith("ray") for d in deps)
