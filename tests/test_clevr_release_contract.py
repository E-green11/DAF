import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import test_release_layout as release_layout


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce_table1_vit_b16_npz_test_best.sh"
MANIFEST = ROOT / "configs" / "clevr_spt_split_sha256.txt"

EXPECTED_CLEVR_HASHES = {
    "vtab-1k/clevr_count/train800val200.txt": "07d728bd0d358f30cda89c898416f4d75beb73d4ba2b6dc064fca569c7e95f68",
    "vtab-1k/clevr_count/val200.txt": "d5a08431e272f078dacd1f29b61e43e21c821ee79a47ea84fcbfd4338835ff62",
    "vtab-1k/clevr_count/test.txt": "f9a6517db88065750f86fffa217d42b456ec54ba5275bfc9e855d119d0c35532",
    "vtab-1k/clevr_dist/train800val200.txt": "d8118b0a313aaccf13eb4f8ab1660119b4106c2b9f4b03041d77d551d4924920",
    "vtab-1k/clevr_dist/val200.txt": "9e954543a12c03e13ead0d864d13624c5122ed263d1be1e07a93f662fb521a1c",
    "vtab-1k/clevr_dist/test.txt": "32596ebb7a4c18f1553bdc778fa23d8d2d8f19c228d624da453950f56b5c2aa1",
}


def _parse_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text().splitlines():
        digest, relative = line.split()
        assert len(digest) == 64
        assert not Path(relative).is_absolute()
        assert relative not in entries
        entries[relative] = digest
    return entries


def _write_fixture_release(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    release = tmp_path / "release"
    script = release / "scripts" / SCRIPT.name
    manifest = release / "configs" / MANIFEST.name
    script.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    script.write_text(SCRIPT.read_text())

    data_root = tmp_path / "ordinary-data"
    clevr_root = tmp_path / "clevr-spt"
    checkpoint = tmp_path / "ViT-B_16.npz"
    checkpoint.touch()
    (data_root / "cifar-100-python").mkdir(parents=True)

    for name, contract in release_layout.EXPECTED_REPRODUCTION_VTAB_CALLS.items():
        if name in {"clevr_count", "clevr_dist"}:
            continue
        task_root = data_root / "vtab-1k" / contract[0]
        task_root.mkdir(parents=True)
        for list_name in {"train800val200.txt", contract[-1], "test.txt"}:
            (task_root / list_name).write_text("fixture\n")

    manifest_lines = []
    for relative in EXPECTED_CLEVR_HASHES:
        payload = f"SPT fixture for {relative}\n"
        target = clevr_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload)
        manifest_lines.append(f"{hashlib.sha256(payload.encode()).hexdigest()}  {relative}")
    manifest.write_text("\n".join(manifest_lines) + "\n")
    return script, data_root, clevr_root, checkpoint


def _preflight_env(data_root: Path, clevr_root: Path, checkpoint: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": sys.executable,
            "DATA_ROOT": str(data_root),
            "CLEVR_DATA_ROOT": str(clevr_root),
            "CHECKPOINT": str(checkpoint),
            "GPU_LIST": "0",
        }
    )
    return env


def test_public_manifest_is_exactly_the_six_original_spt_tfds_split_lists():
    assert _parse_manifest(MANIFEST) == EXPECTED_CLEVR_HASHES


def test_reproduction_snapshot_changes_only_the_two_clevr_data_roots():
    calls = {}
    roots = {}
    for line in SCRIPT.read_text().splitlines():
        if not line.strip().startswith("launch_vtab "):
            continue
        tokens = shlex.split(line)
        name, task, data_root = tokens[1:4]
        calls[name] = (task, *tokens[4:])
        roots[name] = data_root

    assert calls == release_layout.EXPECTED_REPRODUCTION_VTAB_CALLS
    assert roots["clevr_count"] == "$CLEVR_DATA_ROOT"
    assert roots["clevr_dist"] == "$CLEVR_DATA_ROOT"
    assert {root for name, root in roots.items() if not name.startswith("clevr_")} == {"$DATA_ROOT"}


def test_check_accepts_matching_clevr_manifest_from_separate_root(tmp_path):
    script, data_root, clevr_root, checkpoint = _write_fixture_release(tmp_path)
    completed = subprocess.run(
        ["bash", str(script), "--check"],
        cwd=tmp_path,
        env=_preflight_env(data_root, clevr_root, checkpoint),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "preflight OK" in completed.stdout
    assert str(data_root) in completed.stdout
    assert str(clevr_root) in completed.stdout


def test_check_defaults_clevr_to_the_common_data_root(tmp_path):
    script, data_root, clevr_root, checkpoint = _write_fixture_release(tmp_path)
    for task in ("clevr_count", "clevr_dist"):
        source = clevr_root / "vtab-1k" / task
        destination = data_root / "vtab-1k" / task
        source.rename(destination)
    env = _preflight_env(data_root, clevr_root, checkpoint)
    env.pop("CLEVR_DATA_ROOT")

    completed = subprocess.run(
        ["bash", str(script), "--check"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert f"CLEVR data root: {data_root}" in completed.stdout


def test_wrong_clevr_split_hash_stops_before_any_training(tmp_path):
    script, data_root, clevr_root, checkpoint = _write_fixture_release(tmp_path)
    wrong = clevr_root / "vtab-1k/clevr_count/train800val200.txt"
    wrong.write_text("HF-like ordinary CLEVR split\n")

    train_sentinel = tmp_path / "training-started"
    python_wrapper = tmp_path / "python-wrapper"
    python_wrapper.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == -m && \"${2:-}\" == daf_vit.train ]]; then\n"
        f"  touch {shlex.quote(str(train_sentinel))}\n"
        "  exit 0\n"
        "fi\n"
        f"exec {shlex.quote(sys.executable)} \"$@\"\n"
    )
    python_wrapper.chmod(0o755)
    env = _preflight_env(data_root, clevr_root, checkpoint)
    env["PYTHON"] = str(python_wrapper)
    env["OUTPUT_ROOT"] = str(tmp_path / "outputs")
    env["LOG_ROOT"] = str(tmp_path / "logs")

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode != 0
    assert "CLEVR" in completed.stderr
    assert "checksum mismatch" in completed.stderr
    assert "SPT/TFDS" in completed.stderr
    assert "clevr_count/train800val200.txt" in completed.stderr
    assert not train_sentinel.exists()
    assert not Path(env["OUTPUT_ROOT"]).exists()
    assert not Path(env["LOG_ROOT"]).exists()


def test_docs_explain_the_clevr_spt_split_contract_without_claiming_image_hashes():
    docs = [
        ROOT / "docs/DATA_AND_WEIGHTS.md",
        ROOT / "docs/REPRODUCE_TABLE1.md",
        ROOT / "data/README.md",
    ]
    combined = "\n".join(path.read_text() for path in docs)
    for token in ("CLEVR_DATA_ROOT", "SPT", "TFDS", MANIFEST.name):
        assert token in combined
    assert "`CLEVR_DATA_ROOT` defaults" in combined
    assert "to `DATA_ROOT`" in combined
    assert "image checksum" not in combined.casefold()
    assert "image hash" not in combined.casefold()
