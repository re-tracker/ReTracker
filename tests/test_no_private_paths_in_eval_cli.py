from pathlib import Path


def test_eval_cli_has_no_hardcoded_private_checkpoint_paths():
    # Open-source repos should not ship with checkpoint defaults pointing to a
    # developer's private filesystem.
    text = Path("retracker/evaluation/cli.py").read_text(encoding="utf-8")

    assert "/nas/home" not in text
    assert "/nas2/home" not in text
    assert "/input/tdl" not in text
