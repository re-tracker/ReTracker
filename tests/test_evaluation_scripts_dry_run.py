import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)


def _eval_script_path(name: str) -> Path:
    return REPO_ROOT / "scripts" / "evaluation" / name


def test_eval_script_resolves_run_id_checkpoint_and_dry_runs(tmp_path: Path):
    train_out = tmp_path / "training"
    eval_out = tmp_path / "eval"
    data_root = tmp_path / "data"
    dataset_root = data_root / "tracking_val" / "tapvid_val"
    dataset_root.mkdir(parents=True)

    run_id = "my_run_1234"
    ckpt_path = train_out / run_id / "version_shared_ckpt" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("dummy checkpoint")

    env = os.environ.copy()
    env.update(
        {
            # Where evaluation scripts should look for training outputs.
            "RETRACKER_OUTPUT_ROOT": str(train_out),
            # Where evaluation scripts should write outputs.
            "RETRACKER_EVAL_OUTPUT_ROOT": str(eval_out),
            # Default data root (not used in this test since we pass dataset_root explicitly).
            "RETRACKER_DATA_ROOT": str(data_root),
            # Must skip actually running evaluation (datasets/checkpoints aren't real in CI).
            "RETRACKER_EVAL_DRY_RUN": "1",
        }
    )

    script = _eval_script_path("retracker_davis_first.sh")
    proc = _run(["bash", str(script), run_id, str(dataset_root)], env=env)

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert str(ckpt_path.resolve()) in proc.stdout
    assert "DRY RUN" in proc.stdout

    out_dir_line = next(
        line for line in proc.stdout.splitlines() if line.startswith("[eval] Output dir:")
    )
    out_dir = Path(out_dir_line.split(":", 1)[1].strip())
    assert out_dir.is_dir()


def test_eval_script_supports_run_name_prefix_resolution(tmp_path: Path):
    train_out = tmp_path / "training"
    eval_out = tmp_path / "eval"
    dataset_root = tmp_path / "dataset_root"
    dataset_root.mkdir(parents=True)

    # Only one run matches the prefix, so the script should resolve it.
    run_id = "expname_abcd1234"
    ckpt_path = train_out / run_id / "version_shared_ckpt" / "last.ckpt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.write_text("dummy checkpoint")

    env = os.environ.copy()
    env.update(
        {
            "RETRACKER_OUTPUT_ROOT": str(train_out),
            "RETRACKER_EVAL_OUTPUT_ROOT": str(eval_out),
            "RETRACKER_EVAL_DRY_RUN": "1",
        }
    )

    script = _eval_script_path("retracker_davis_first.sh")
    proc = _run(["bash", str(script), "expname", str(dataset_root)], env=env)

    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert str(ckpt_path.resolve()) in proc.stdout
