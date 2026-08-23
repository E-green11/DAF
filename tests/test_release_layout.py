import re
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_COPIED_FILES = frozenset(
    {
        "src/daf_vit/__init__.py",
        "src/daf_vit/data.py",
        "src/daf_vit/dynamic.py",
        "src/daf_vit/lora.py",
        "src/daf_vit/selection.py",
        "src/daf_vit/train.py",
        "tests/test_data.py",
        "tests/test_data_filelist.py",
        "tests/test_dynamic.py",
        "tests/test_export_vtab1k_hf.py",
        "tests/test_lora.py",
        "tests/test_selection.py",
        "tests/test_sparse_lora.py",
        "tests/test_summarize_table1_vtab.py",
        "tests/test_train_modes.py",
        "scripts/export_vtab1k_hf.py",
        "scripts/export_vtab1k_spt.py",
        "scripts/prepare_vtab1k_hf.sh",
        "scripts/prepare_vtab1k_spt.sh",
        "scripts/reproduce_table1_vit_b16_npz_test_best.sh",
        "scripts/summarize_table1_vtab.py",
        "scripts/train_cifar100_daf_vit_b16_npz.sh",
        "scripts/train_cifar100_head_only_augreg.sh",
        "scripts/train_table1_vtab_daf.sh",
        "scripts/train_vtab19.sh",
        "scripts/train_vtab_cifar100.sh",
        "scripts/tune_structured_testbest_stage1.sh",
        "pyproject.toml",
        "requirements.txt",
        "DAF_framework.png",
        "docs/table1_datasets.md",
    }
)
REQUIRED_RELEASE_ADDITIONS = frozenset(
    {
        "configs/clevr_spt_split_sha256.txt",
        "requirements-repro.txt",
        "requirements-spt.txt",
        "tests/test_clevr_release_contract.py",
        "tests/test_export_vtab1k_spt.py",
        "tests/test_launcher_pid_safety.py",
    }
)
LEGACY_DATA_DIR_NAME = "data" + "-formal"
FORBIDDEN_NAMES = {".git", "__pycache__", ".pytest_cache", "outputs", "logs", LEGACY_DATA_DIR_NAME}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".npz",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".npy",
    ".pkl",
    ".h5",
    ".log",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".tgz",
}
ASSET_READMES = {Path("data/README.md"), Path("checkpoints/README.md")}
PUBLIC_DOCUMENT_SUFFIXES = {".md", ".rst", ".txt", ".csv", ".json"}
HISTORICAL_RESULT_FILENAME = re.compile(
    r"(?:^|[_-])(?:historical[_-]?)?(?:results|accuracies|metrics|scores|leaderboard|benchmark)(?:[_-]|$)",
    re.IGNORECASE,
)
HISTORICAL_RESULT_CONTENT = re.compile(
    r"(?:\b(?:results?|accuracies|scores|metrics?)\b.{0,240}"
    r"\b(?:accuracy|acc1|top[- ]?1|macro|mean)\b.{0,80}\b\d+(?:\.\d+)?\s*%?"
    r"|\btable[- ]?1\b.{0,160}\b(?:accuracy|acc1|top[- ]?1|score)\b.{0,80}\b\d+(?:\.\d+)?\s*%?)",
    re.IGNORECASE | re.DOTALL,
)

EXPECTED_REPRODUCTION_VTAB_CALLS = {
    "caltech101": ("caltech101", "120", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.485,0.456,0.406", "0.229,0.224,0.225", "test.txt"),
    "dtd": ("dtd", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "oxford_flowers102": ("oxford_flowers102", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "oxford_iiit_pet": ("oxford_iiit_pet", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "svhn": ("svhn", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "sun397": ("sun397", "120", "0.001", "0.1", "0.0", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "test.txt"),
    "eurosat": ("eurosat", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "resisc45": ("resisc45", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "diabetic_retinopathy": ("diabetic_retinopathy", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "patch_camelyon": ("patch_camelyon", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "clevr_count": ("clevr_count", "200", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "clevr_dist": ("clevr_dist", "200", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "dmlab": ("dmlab", "200", "0.001", "0.1", "0.1", "8", "16", "0.0", "category", "grad_square", "0", "random-resized-crop", "resize-center-crop", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "kitti": ("kitti", "200", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "dsprites_loc": ("dsprites_loc", "200", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "dsprites_ori": ("dsprites_ori", "200", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "smallnorb_azi": ("smallnorb_azi", "200", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
    "smallnorb_ele": ("smallnorb_ele", "200", "0.001", "0.0", "0.1", "8", "8", "0.1", "global", "grad_weight_abs", "1", "direct-resize", "direct-resize", "0.5,0.5,0.5", "0.5,0.5,0.5", "val200.txt"),
}

BACKGROUND_LAUNCH_CASES = (
    ("reproduce_table1_vit_b16_npz_test_best.sh", "dtd", 19),
    ("train_table1_vtab_daf.sh", "cifar", 1),
    ("tune_structured_testbest_stage1.sh", "dsprites_ori_direct_cat_gs_lr1e3", 3),
)
GPU_LAUNCHERS = (
    "reproduce_table1_vit_b16_npz_test_best.sh",
    "train_cifar100_daf_vit_b16_npz.sh",
    "train_cifar100_head_only_augreg.sh",
    "train_table1_vtab_daf.sh",
    "train_vtab_cifar100.sh",
    "tune_structured_testbest_stage1.sh",
)


def _is_internal_workflow_path(relative: Path) -> bool:
    return relative.parts[:2] == ("docs", "superpowers")


def _is_public_document(relative: Path, path: Path) -> bool:
    if _is_internal_workflow_path(relative):
        return False
    if relative.parts and relative.parts[0] in {"src", "scripts", "tests"}:
        return False
    return path.suffix.lower() in PUBLIC_DOCUMENT_SUFFIXES


def _public_metadata_files() -> list[Path]:
    suffixes = PUBLIC_DOCUMENT_SUFFIXES | {".yaml", ".yml", ".toml"}
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or (path.suffix.lower() not in suffixes and path.name != ".gitignore"):
            continue
        relative = path.relative_to(ROOT)
        if _is_internal_workflow_path(relative):
            continue
        if relative.parts and relative.parts[0] in {"src", "scripts", "tests"}:
            continue
        paths.append(path)
    return sorted(paths)


def _write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import os
import signal
import sys
import time
from pathlib import Path


def append_event(value):
    with Path(os.environ["FAKE_EVENT_LOG"]).open("a") as handle:
        handle.write(value + "\\n")


args = sys.argv[1:]
if args[:2] == ["-m", "daf_vit.train"]:
    train_args = args[2:]
    try:
        output_dir = Path(train_args[train_args.index("--output-dir") + 1])
    except (ValueError, IndexError):
        raise SystemExit(90)
    time.sleep(float(os.environ.get("FAKE_PRECREATE_SLEEP", "0")))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = output_dir.name

    def terminate(signum, _frame):
        append_event(f"terminate:{run_name}:{os.getpid()}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    append_event(f"start:{run_name}:{os.getpid()}")
    if run_name == os.environ.get("FAIL_TASK"):
        append_event(f"fail:{run_name}:{os.getpid()}")
        raise SystemExit(23)
    time.sleep(float(os.environ.get("FAKE_SLEEP", "0.02")))
    append_event(f"finish:{run_name}:{os.getpid()}")
    raise SystemExit(0)
if args and args[0] == "-" and len(args) > 1 and args[1].endswith("/args.json"):
    print(0)
    raise SystemExit(0)
if args and args[0] == "-" and len(args) > 2 and args[1].endswith("clevr_spt_split_sha256.txt"):
    # The exact checksum contract is exercised with a real interpreter in
    # test_clevr_release_contract.py. Launcher lifecycle tests only need this
    # helper to stand in for that already-validated preflight step.
    raise SystemExit(0)
if args and (args[0] == "-" or args[0].endswith("/summarize_table1_vtab.py")):
    Path(os.environ["SUMMARY_SENTINEL"]).touch()
    raise SystemExit(0)
raise SystemExit(91)
"""
    )
    fake_python.chmod(0o755)
    return fake_python


def _write_required_lists(task_root: Path, names: set[str]) -> None:
    task_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (task_root / name).touch()


def _launcher_test_environment(tmp_path: Path, script_name: str) -> tuple[dict[str, str], Path, Path, int]:
    fake_python = _write_fake_python(tmp_path)
    data_root = tmp_path / "external-data"
    clevr_root = data_root / "clevr-spt"
    checkpoint = tmp_path / "weights" / "ViT-B_16.npz"
    output_root = tmp_path / "run-outputs"
    log_root = tmp_path / "run-logs"
    event_log = tmp_path / "events.log"
    summary_sentinel = tmp_path / "summary-called"
    checkpoint.parent.mkdir()
    checkpoint.touch()

    expected_jobs = 0
    if script_name == "reproduce_table1_vit_b16_npz_test_best.sh":
        (data_root / "cifar-100-python").mkdir(parents=True)
        clevr_root = data_root / "clevr-spt"
        for name, contract in EXPECTED_REPRODUCTION_VTAB_CALLS.items():
            task = contract[0]
            val_list = contract[-1]
            task_data_root = clevr_root if name in {"clevr_count", "clevr_dist"} else data_root
            _write_required_lists(
                task_data_root / "vtab-1k" / task,
                {"train800val200.txt", val_list, "test.txt"},
            )
        env_clevr_root = str(clevr_root)
        expected_jobs = 19
    elif script_name == "train_table1_vtab_daf.sh":
        _write_required_lists(
            data_root / "vtab-1k" / "cifar",
            {"train800val200.txt", "val200.txt", "test.txt"},
        )
        expected_jobs = 1
    elif script_name == "tune_structured_testbest_stage1.sh":
        for task in ("dsprites_loc", "dsprites_ori", "clevr_dist"):
            _write_required_lists(
                data_root / "vtab-1k" / task,
                {"train800val200.txt", "val200.txt", "test.txt"},
            )
        expected_jobs = 3
    else:
        raise AssertionError(f"unsupported test launcher: {script_name}")

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "DATA_ROOT": str(data_root),
            "CHECKPOINT": str(checkpoint),
            "OUTPUT_ROOT": str(output_root),
            "LOG_ROOT": str(log_root),
            "GPU_LIST": "0,1,2,3",
            "FAKE_EVENT_LOG": str(event_log),
            "SUMMARY_SENTINEL": str(summary_sentinel),
        }
    )
    if script_name == "reproduce_table1_vit_b16_npz_test_best.sh":
        env["CLEVR_DATA_ROOT"] = env_clevr_root
    if script_name == "train_table1_vtab_daf.sh":
        env["TASKS"] = "cifar"
    return env, event_log, summary_sentinel, expected_jobs


def _wait_for_event(event_log: Path, prefix: str, timeout: float = 3.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if event_log.exists():
            for line in event_log.read_text().splitlines():
                if line.startswith(prefix):
                    return line
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {prefix!r} in {event_log}")


def test_release_tree(root: Path) -> list[str]:
    """Return release artifacts that must not be included in a source-only upload."""
    violations = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            violations.append(f"forbidden path: {relative}")
        if path.is_symlink():
            violations.append(f"symbolic link: {relative}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden suffix: {relative}")
        if relative.parts and relative.parts[0] in {"data", "checkpoints"}:
            if path.is_file() and relative not in ASSET_READMES:
                violations.append(f"external asset file: {relative}")
        if path.is_file() and _is_public_document(relative, path):
            if HISTORICAL_RESULT_FILENAME.search(path.stem):
                violations.append(f"historical result filename: {relative}")
            text = path.read_text(errors="ignore")
            if HISTORICAL_RESULT_CONTENT.search(text):
                violations.append(f"historical result content: {relative}")
    return sorted(set(violations))


test_release_tree.__test__ = False


def test_required_release_files_exist():
    assert len(REQUIRED_COPIED_FILES) == 31
    required = REQUIRED_COPIED_FILES | REQUIRED_RELEASE_ADDITIONS
    missing = sorted(path for path in required if not (ROOT / path).is_file())
    assert not missing, f"missing release files: {missing}"


def test_release_tree_has_no_external_artifacts():
    violations = test_release_tree(ROOT)
    assert not violations, f"forbidden release artifacts: {violations}"


def test_public_files_have_no_machine_specific_paths():
    paths = set(_public_metadata_files())
    for directory in (ROOT / "scripts", ROOT / "src"):
        paths.update(path for path in directory.rglob("*") if path.is_file())
    offenders = []
    machine_markers = ("/mnt/" + "sda/", "/" + "home/", "DAF " + "copy")
    legacy_marker = "data-" + "formal"
    for path in sorted(paths):
        text = path.read_text(errors="ignore")
        has_machine_path = any(marker in text for marker in machine_markers)
        has_public_legacy_path = legacy_marker in text and path.name != ".gitignore"
        if has_machine_path or has_public_legacy_path:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"machine-specific or legacy paths: {offenders}"


def test_public_metadata_scan_covers_root_and_nested_metadata():
    relative_paths = {str(path.relative_to(ROOT)) for path in _public_metadata_files()}
    expected = {
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-repro.txt",
        "requirements-spt.txt",
        "configs/clevr_spt_split_sha256.txt",
        "data/README.md",
        "checkpoints/README.md",
        "docs/DATA_AND_WEIGHTS.md",
        "docs/REPRODUCE_TABLE1.md",
        "docs/table1_datasets.md",
    }
    assert expected <= relative_paths
    assert not any(path.startswith("docs/superpowers/") for path in relative_paths)


def test_release_documentation_covers_runtime_contract():
    required_docs = {
        "README.md",
        "docs/DATA_AND_WEIGHTS.md",
        "docs/REPRODUCE_TABLE1.md",
        "checkpoints/README.md",
        "data/README.md",
        "requirements-repro.txt",
        "requirements-spt.txt",
        "configs/clevr_spt_split_sha256.txt",
    }
    missing = sorted(path for path in required_docs if not (ROOT / path).is_file())
    assert not missing, f"missing operational documents: {missing}"

    readme = (ROOT / "README.md").read_text()
    data_doc = (ROOT / "docs/DATA_AND_WEIGHTS.md").read_text()
    run_doc = (ROOT / "docs/REPRODUCE_TABLE1.md").read_text()
    for token in (
        "Python 3.10",
        "python -m pip install -r requirements.txt",
        "python -m pytest tests -q",
        "--check",
        "GPU_LIST=0 bash scripts/reproduce_table1_vit_b16_npz_test_best.sh",
        "GPU_LIST=0,1,2,3 bash scripts/reproduce_table1_vit_b16_npz_test_best.sh",
        "docs/DATA_AND_WEIGHTS.md",
        "docs/REPRODUCE_TABLE1.md",
    ):
        assert token in readme, f"README missing runtime instruction: {token}"
    for token in ("checkpoints/ViT-B_16.npz", "data/cifar-100-python", "data/vtab-1k"):
        assert token in data_doc, f"data documentation missing layout token: {token}"
    for token in (
        "DATA_ROOT",
        "CLEVR_DATA_ROOT",
        "PYTHON",
        "CHECKPOINT",
        "OUTPUT_ROOT",
        "LOG_ROOT",
        "GPU_LIST",
        "NUM_WORKERS",
        "BATCH_SIZE",
        "SEED",
        "MODEL",
        "EVAL_INTERVAL",
        "SAVE_EVERY",
        ".complete",
        ".daf-launch.lock",
    ):
        assert token in run_doc, f"reproduction documentation missing contract token: {token}"


def test_reproducibility_dependency_lock_matches_validated_environment():
    from pip._vendor.packaging.requirements import Requirement

    text = (ROOT / "requirements-repro.txt").read_text()
    expected = {
        "timm==1.0.28",
        "PyYAML==6.0.3",
        "tqdm==4.69.0",
        "pytest==9.1.1",
        "datasets==5.0.0",
        "huggingface-hub==1.24.0",
        "numpy==1.26.4",
    }
    lines = {line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")}
    assert lines == expected
    for requirement in lines:
        Requirement(requirement)

    readme = (ROOT / "README.md").read_text()
    assert "torch==2.11.0+cu128 torchvision==0.26.0+cu128" in readme
    assert "--index-url https://download.pytorch.org/whl/cu128" in readme
    assert "python -m pip install -r requirements-repro.txt" in readme

    optional_spt = {
        line.strip()
        for line in (ROOT / "requirements-spt.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert {
        "tensorflow-cpu==2.15.1",
        "keras==2.15.0",
        "tensorflow-addons==0.23.0",
        "tensorflow-datasets==4.9.4",
        "tensorflow-metadata==1.15.0",
        "protobuf==3.20.3",
        "Pillow==10.2.0",
    } == optional_spt
    for requirement in optional_spt:
        Requirement(requirement)


def test_dataset_documentation_lists_all_runtime_tasks_and_lists():
    text = (ROOT / "docs/DATA_AND_WEIGHTS.md").read_text()
    expected_tasks = {contract[0] for contract in EXPECTED_REPRODUCTION_VTAB_CALLS.values()}
    assert len(expected_tasks) == 18
    assert "data/cifar-100-python" in text
    row_pattern = re.compile(
        r"^\| [^|]+ \| `(?P<root>data/vtab-1k|\$CLEVR_DATA_ROOT/vtab-1k)/(?P<task>[a-z0-9_]+)/` "
        r"\| `(?P<train>[^`]+)` \| `(?P<val>[^`]+)` \| `(?P<test>[^`]+)` \|$",
        re.MULTILINE,
    )
    rows = {match.group("task"): match.groupdict() for match in row_pattern.finditer(text)}
    assert set(rows) == expected_tasks
    for task, contract in EXPECTED_REPRODUCTION_VTAB_CALLS.items():
        row = rows[task]
        assert row["root"] == "data/vtab-1k"
        assert row["train"] == "train800val200.txt"
        assert row["val"] == contract[-1]
        assert row["test"] == "test.txt"
    assert "16 non-CLEVR" in text
    assert "two CLEVR" in text
    assert "Neither prepares CIFAR-100" in text
    assert "CHECKPOINT" in text


def test_public_docs_do_not_publish_historical_scores_or_selection_policy():
    public_docs = _public_metadata_files()
    forbidden_terms = (
        "76" + ".95",
        "Paper " + "DAF",
        "Per-dataset " + "comparison",
        "historical " + "accuracy",
        "historical " + "aggregate",
        "highest-observed-" + "test",
        "highest observed " + "test",
        "test" + ".acc1",
    )
    offenders = []
    for path in public_docs:
        text = path.read_text(errors="ignore")
        if any(term.casefold() in text.casefold() for term in forbidden_terms):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"public documents expose historical scores or selection policy: {offenders}"


def test_public_dataset_docs_have_no_legacy_or_unsupported_tasks():
    offenders = []
    unsupported_pattern = re.compile(
        r"\b(?:cub|nabirds|oxfordflower|stanforddog|stanfordcar)\b", re.IGNORECASE
    )
    for path in _public_metadata_files():
        text = path.read_text(errors="ignore")
        if "vtab-1k/cifar" in text or re.search(r"\btrain800\.txt\b", text) or unsupported_pattern.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"legacy or unsupported dataset instructions: {offenders}"


@pytest.mark.parametrize("script_name", ("prepare_vtab1k_hf.sh", "prepare_vtab1k_spt.sh"))
def test_preparation_helpers_default_to_their_exact_reproduction_subsets(tmp_path, script_name):
    argv_path = tmp_path / "python-argv.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$PREPARE_ARGV_PATH\"\n"
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PYTHON": str(fake_python),
            "DATA_ROOT": str(tmp_path / "data"),
            "PREPARE_ARGV_PATH": str(argv_path),
        }
    )
    if script_name == "prepare_vtab1k_spt.sh":
        spt_root = tmp_path / "SPT"
        source = spt_root / "data" / "vtab-source" / "get_vtab1k.py"
        source.parent.mkdir(parents=True)
        source.touch()
        env["SPT_ROOT"] = str(spt_root)

    script = ROOT / "scripts" / script_name
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    argv = argv_path.read_text().splitlines()
    assert "--tasks" in argv
    tasks = argv[argv.index("--tasks") + 1 :]
    all_tasks = {contract[0] for contract in EXPECTED_REPRODUCTION_VTAB_CALLS.values()}
    expected = {"clevr_count", "clevr_dist"}
    if script_name == "prepare_vtab1k_hf.sh":
        expected = all_tasks - expected
    assert len(tasks) == len(expected)
    assert set(tasks) == expected
    assert "cifar" not in tasks
    if script_name == "prepare_vtab1k_spt.sh":
        output_root = argv[argv.index("--output-root") + 1]
        assert output_root == str(Path(env["DATA_ROOT"]) / "vtab-1k")

    help_result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert help_result.returncode == 0
    assert str(len(expected)) in help_result.stdout and "TASKS" in help_result.stdout

    override_env = env.copy()
    override_env["TASKS"] = "dtd sun397"
    override_result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=override_env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert override_result.returncode == 0, override_result.stderr
    override_argv = argv_path.read_text().splitlines()
    assert override_argv[override_argv.index("--tasks") + 1 :] == ["dtd", "sun397"]


def test_shell_launchers_have_portable_defaults_and_help(tmp_path):
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    missing_contract = []
    for path in scripts:
        text = path.read_text()
        required = ('SCRIPT_DIR=', 'REPO_ROOT=', 'PYTHON=${PYTHON:-python}', '--help')
        if any(token not in text for token in required):
            missing_contract.append(path.name)
    assert not missing_contract, f"launchers missing portable/help contract: {missing_contract}"

    for path in scripts:
        completed = subprocess.run(
            ["bash", str(path), "--help"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert completed.returncode == 0, f"{path.name} --help failed: {completed.stderr}"
        assert "Usage:" in completed.stdout


def test_reproduction_launcher_preserves_the_19_task_snapshot():
    path = ROOT / "scripts/reproduce_table1_vit_b16_npz_test_best.sh"
    text = path.read_text()
    calls = {}
    data_roots = {}
    for line in text.splitlines():
        if not line.strip().startswith("launch_vtab "):
            continue
        tokens = shlex.split(line.strip())
        assert len(tokens) == 19, f"unexpected launch_vtab fields: {tokens}"
        name, task, data_root = tokens[1:4]
        assert name not in calls, f"duplicate task launch: {name}"
        calls[name] = (task, *tokens[4:])
        data_roots[name] = data_root

    assert text.splitlines().count("launch_cifar100") == 1
    assert len(calls) + 1 == 19
    assert calls == EXPECTED_REPRODUCTION_VTAB_CALLS
    assert data_roots["clevr_count"] == "$CLEVR_DATA_ROOT"
    assert data_roots["clevr_dist"] == "$CLEVR_DATA_ROOT"
    assert {
        root for task, root in data_roots.items() if task not in {"clevr_count", "clevr_dist"}
    } == {"$DATA_ROOT"}

    assert "MODEL=${MODEL:-vit_base_patch16_224}" in text
    assert "CHECKPOINT=${CHECKPOINT:-$REPO_ROOT/checkpoints/ViT-B_16.npz}" in text
    assert "BATCH_SIZE=${BATCH_SIZE:-64}" in text
    assert "SEED=${SEED:-0}" in text
    cifar_contract = {
        "--dataset cifar100",
        "--epochs 200",
        "--train-size 800",
        "--val-size 200",
        "--lr 0.0003",
        "--weight-decay 0.0001",
        "--warmup-epochs 10",
        "--rank 8",
        "--lora-alpha 16",
        "--budget 0.2",
        "--update-interval 10",
        "--sensitivity-batches 32",
        "--sensitivity-score-mode grad_square",
    }
    missing = sorted(token for token in cifar_contract if token not in text)
    assert not missing, f"CIFAR-100 reproduction contract changed: {missing}"

    common_body = text.split("launch_common() {", 1)[1].split("\n}", 1)[0]
    shared_training_contract = {
        '--batch-size "$BATCH_SIZE"',
        '--seed "$SEED"',
        '--num-workers "$NUM_WORKERS"',
        '--eval-interval "$EVAL_INTERVAL"',
        '--save-every "$SAVE_EVERY"',
        '--method daf',
        '--image-size 224',
        '--device cuda',
        '--amp',
    }
    missing = sorted(token for token in shared_training_contract if token not in common_body)
    assert not missing, f"shared reproduction contract changed: {missing}"

    vtab_body = text.split("launch_vtab() {", 1)[1].split("\n}", 1)[0]
    vtab_common_contract = {
        '--dataset "vtab:$task"',
        '--epochs "$epochs"',
        '--weight-decay 0.0001',
        '--warmup-epochs 10',
        '--budget 0.2',
        '--update-interval 10',
        '--sensitivity-batches 16',
        '--vtab-train-list train800val200.txt',
        '--vtab-val-list "$val_list"',
        '--vtab-test-list test.txt',
    }
    missing = sorted(token for token in vtab_common_contract if token not in vtab_body)
    assert not missing, f"common VTAB reproduction contract changed: {missing}"
    assert "--test-on-best-val-only" not in text
    assert "--accum" not in text


def test_reproduction_launcher_uses_portable_paths_and_current_run_summary():
    text = (ROOT / "scripts/reproduce_table1_vit_b16_npz_test_best.sh").read_text()
    required = {
        "PYTHON=${PYTHON:-python}",
        "DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data}",
        "CLEVR_DATA_ROOT=${CLEVR_DATA_ROOT:-$DATA_ROOT}",
        "OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs/reproduce_table1_vit_b16_npz_test_best}",
        "LOG_ROOT=${LOG_ROOT:-$REPO_ROOT/logs/reproduce_table1_vit_b16_npz_test_best}",
        'PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"',
        'row.get("test", {}).get("acc1")',
        "if best is None or item[0] > best[0]",
    }
    missing = sorted(token for token in required if token not in text)
    assert not missing, f"missing reproduction behavior: {missing}"
    assert LEGACY_DATA_DIR_NAME not in text
    assert "paper =" not in text
    assert "Paper DAF" not in text
    assert "| Diff |" not in text


def test_reproduction_preflight_can_check_external_layout_from_any_cwd(tmp_path):
    readme = (ROOT / "README.md").read_text()
    run_doc = (ROOT / "docs/REPRODUCE_TABLE1.md").read_text()
    for text in (readme, run_doc):
        assert "Run relative commands from the repository root" in text
        assert "bash /path/to/DAFgit/scripts/reproduce_table1_vit_b16_npz_test_best.sh --check" in text

    data_root = tmp_path / "external-data"
    clevr_root = data_root / "clevr-spt"
    checkpoint = tmp_path / "weights" / "ViT-B_16.npz"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    (data_root / "cifar-100-python").mkdir(parents=True)
    for name, contract in EXPECTED_REPRODUCTION_VTAB_CALLS.items():
        task = contract[0]
        val_list = contract[-1]
        task_data_root = clevr_root if name in {"clevr_count", "clevr_dist"} else data_root
        task_root = task_data_root / "vtab-1k" / task
        task_root.mkdir(parents=True)
        for list_name in {"train800val200.txt", val_list, "test.txt"}:
            (task_root / list_name).touch()

    env = os.environ.copy()
    env.update(
        {
            "PYTHON": "/bin/true",
            "DATA_ROOT": str(data_root),
            "CLEVR_DATA_ROOT": str(clevr_root),
            "CHECKPOINT": str(checkpoint),
            "GPU_LIST": "0,1",
        }
    )
    script = ROOT / "scripts/reproduce_table1_vit_b16_npz_test_best.sh"
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
    assert "preflight OK" in completed.stdout
    assert not (ROOT / "outputs").exists()
    assert not (ROOT / "logs").exists()


@pytest.mark.parametrize("script_name,failed_run,expected_jobs", BACKGROUND_LAUNCH_CASES)
def test_background_launchers_propagate_failure_without_summarizing(
    tmp_path, script_name, failed_run, expected_jobs
):
    env, event_log, summary_sentinel, configured_jobs = _launcher_test_environment(tmp_path, script_name)
    assert configured_jobs == expected_jobs
    env["FAIL_TASK"] = failed_run
    script = ROOT / "scripts" / script_name

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode != 0
    assert not summary_sentinel.exists(), "summary must not run after a failed training job"
    assert not (Path(env["OUTPUT_ROOT"]) / failed_run / ".complete").exists()
    events = event_log.read_text().splitlines()
    started_jobs = min(expected_jobs, 4)
    assert len([line for line in events if line.startswith("start:")]) == started_jobs
    assert len([line for line in events if line.startswith(("finish:", "fail:"))]) == started_jobs


@pytest.mark.parametrize("script_name,_failed_run,expected_jobs", BACKGROUND_LAUNCH_CASES)
def test_background_launchers_mark_only_successful_complete_runs(
    tmp_path, script_name, _failed_run, expected_jobs
):
    env, event_log, summary_sentinel, configured_jobs = _launcher_test_environment(tmp_path, script_name)
    assert configured_jobs == expected_jobs
    script = ROOT / "scripts" / script_name

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    if script_name != "tune_structured_testbest_stage1.sh":
        assert summary_sentinel.is_file()
    assert len(list(Path(env["OUTPUT_ROOT"]).glob("*/.complete"))) == expected_jobs
    events = event_log.read_text().splitlines()
    assert len([line for line in events if line.startswith("finish:")]) == expected_jobs


@pytest.mark.parametrize(
    "script_name,run_name,_expected_jobs",
    (
        ("reproduce_table1_vit_b16_npz_test_best.sh", "cifar100", 19),
        ("train_table1_vtab_daf.sh", "cifar", 1),
        ("tune_structured_testbest_stage1.sh", "dsprites_loc_direct_cat_gs_lr1e3", 3),
    ),
)
def test_background_launchers_refuse_residual_output_without_complete_marker(
    tmp_path, script_name, run_name, _expected_jobs
):
    env, event_log, summary_sentinel, _ = _launcher_test_environment(tmp_path, script_name)
    residual = Path(env["OUTPUT_ROOT"]) / run_name
    residual.mkdir(parents=True)
    (residual / "metrics.jsonl").write_text("partial\n")
    script = ROOT / "scripts" / script_name

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode != 0
    assert "without .complete" in completed.stderr
    assert not event_log.exists()
    assert not summary_sentinel.exists()
    assert (residual / "metrics.jsonl").read_text() == "partial\n"


@pytest.mark.parametrize("script_name", GPU_LAUNCHERS)
def test_explicitly_empty_gpu_list_is_rejected(tmp_path, script_name):
    fake_python = _write_fake_python(tmp_path)
    env = os.environ.copy()
    env.update({"PYTHON": str(fake_python), "GPU_LIST": ""})
    script = ROOT / "scripts" / script_name
    text = script.read_text()
    assert "GPU_LIST=${GPU_LIST-0}" in text

    completed = subprocess.run(
        ["bash", str(script), "--check"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode != 0
    assert "GPU_LIST" in completed.stderr


@pytest.mark.parametrize(
    "script_name,late_run",
    (
        ("reproduce_table1_vit_b16_npz_test_best.sh", "dtd"),
        ("train_table1_vtab_daf.sh", "dtd"),
        ("tune_structured_testbest_stage1.sh", "dsprites_ori_direct_cat_gs_lr1e3"),
    ),
)
def test_background_launchers_preflight_all_run_artifacts_before_starting(tmp_path, script_name, late_run):
    env, event_log, summary_sentinel, _ = _launcher_test_environment(tmp_path, script_name)
    if script_name == "train_table1_vtab_daf.sh":
        _write_required_lists(
            Path(env["DATA_ROOT"]) / "vtab-1k" / "dtd",
            {"train800val200.txt", "val200.txt", "test.txt"},
        )
        env["TASKS"] = "cifar dtd"
    residual = Path(env["OUTPUT_ROOT"]) / late_run
    residual.mkdir(parents=True)
    (residual / "metrics.jsonl").write_text("partial\n")
    script = ROOT / "scripts" / script_name

    completed = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode != 0
    assert "without .complete" in completed.stderr
    assert "launch " not in completed.stdout
    assert not event_log.exists(), "no training may start before every run artifact is preflighted"
    assert not summary_sentinel.exists()
    assert not (Path(env["OUTPUT_ROOT"]) / ".daf-launch.lock").exists()
    assert not (Path(env["LOG_ROOT"]) / ".daf-launch.lock").exists()


def test_concurrent_launchers_with_shared_roots_have_single_owner(tmp_path):
    script_name = "train_table1_vtab_daf.sh"
    env, event_log, _summary_sentinel, _ = _launcher_test_environment(tmp_path, script_name)
    env.update(
        {
            "FAKE_PRECREATE_SLEEP": "0.2",
            "FAKE_SLEEP": "0.2",
            "LOG_ROOT": env["OUTPUT_ROOT"],
        }
    )
    script = ROOT / "scripts" / script_name

    first = subprocess.Popen(
        ["bash", str(script)], cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    second = subprocess.Popen(
        ["bash", str(script)], cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    first_stdout, first_stderr = first.communicate(timeout=10)
    second_stdout, second_stderr = second.communicate(timeout=10)

    returncodes = (first.returncode, second.returncode)
    assert sum(code == 0 for code in returncodes) == 1, (
        returncodes,
        first_stdout,
        first_stderr,
        second_stdout,
        second_stderr,
    )
    assert sum(code != 0 for code in returncodes) == 1
    assert "launcher lock" in (first_stderr + second_stderr)
    events = event_log.read_text().splitlines()
    assert len([line for line in events if line.startswith("start:")]) == 1
    assert len(list(Path(env["OUTPUT_ROOT"]).glob("*/.complete"))) == 1
    assert not (Path(env["OUTPUT_ROOT"]) / ".daf-launch.lock").exists()
    assert not (Path(env["LOG_ROOT"]) / ".daf-launch.lock").exists()


@pytest.mark.parametrize("script_name,_failed_run,_expected_jobs", BACKGROUND_LAUNCH_CASES)
def test_launcher_signal_cleanup_terminates_owned_children_and_releases_locks(
    tmp_path, script_name, _failed_run, _expected_jobs
):
    env, event_log, _summary_sentinel, _ = _launcher_test_environment(tmp_path, script_name)
    env["FAKE_SLEEP"] = "5"
    script = ROOT / "scripts" / script_name
    process = subprocess.Popen(
        ["bash", str(script)], cwd=tmp_path, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    start = _wait_for_event(event_log, "start:")
    child_pid = int(start.rsplit(":", 1)[1])

    process.terminate()
    stdout, stderr = process.communicate(timeout=10)
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        child_alive = False
    else:
        child_alive = True
        os.kill(child_pid, signal.SIGTERM)

    assert process.returncode != 0, (stdout, stderr)
    assert not child_alive, f"launcher left owned child running: {child_pid}"
    assert any(line.startswith("terminate:") for line in event_log.read_text().splitlines())
    assert not (Path(env["OUTPUT_ROOT"]) / ".daf-launch.lock").exists()
    assert not (Path(env["LOG_ROOT"]) / ".daf-launch.lock").exists()
