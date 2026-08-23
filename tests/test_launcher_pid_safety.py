import os
import signal
import subprocess
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPRO_TASKS = (
    "caltech101",
    "dtd",
    "oxford_flowers102",
    "oxford_iiit_pet",
    "svhn",
    "sun397",
    "eurosat",
    "resisc45",
    "diabetic_retinopathy",
    "patch_camelyon",
    "clevr_count",
    "clevr_dist",
    "dmlab",
    "kitti",
    "dsprites_loc",
    "dsprites_ori",
    "smallnorb_azi",
    "smallnorb_ele",
)


def _wait_for(path: Path, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    contents = path.read_text() if path.exists() else "<missing>"
    raise AssertionError(f"timed out waiting on {path}: {contents}")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _pid_is_stopped(pid: int) -> bool:
    status = Path(f"/proc/{pid}/status")
    if not status.exists():
        return False
    state = next(line for line in status.read_text().splitlines() if line.startswith("State:"))
    return "T (stopped)" in state or "t (tracing stop)" in state


def _write_list_files(task_root: Path) -> None:
    task_root.mkdir(parents=True, exist_ok=True)
    for name in ("train800val200.txt", "val200.txt", "test.txt"):
        (task_root / name).touch()


def _write_fake_python(tmp_path: Path) -> Path:
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import os
import signal
import sys
import time
from pathlib import Path


def record(event):
    with Path(os.environ["JOB_EVENT_LOG"]).open("a") as handle:
        handle.write(f"{event}:{os.getpid()}\\n")


args = sys.argv[1:]
if args and args[0] == "-":
    raise SystemExit(0)
if args[:2] == ["-m", "daf_vit.train"]:
    train_args = args[2:]
    output_dir = Path(train_args[train_args.index("--output-dir") + 1])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = output_dir.name

    def terminate(signum, _frame):
        record(f"terminate:{run_name}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    record(f"start:{run_name}")
    if run_name == os.environ["FAST_RUN"]:
        record(f"finish:{run_name}")
        raise SystemExit(0)
    time.sleep(30)
    record(f"finish:{run_name}")
    raise SystemExit(0)
raise SystemExit(91)
"""
    )
    fake_python.chmod(0o755)
    return fake_python


def _launcher_environment(tmp_path: Path, script_name: str) -> tuple[dict[str, str], str, str]:
    fake_python = _write_fake_python(tmp_path)
    data_root = tmp_path / "data"
    checkpoint = tmp_path / "weights.npz"
    checkpoint.touch()

    if script_name == "reproduce_table1_vit_b16_npz_test_best.sh":
        (data_root / "cifar-100-python").mkdir(parents=True)
        for task in REPRO_TASKS:
            _write_list_files(data_root / "vtab-1k" / task)
        fast_run = "cifar100"
        slow_run = "caltech101"
    elif script_name == "train_table1_vtab_daf.sh":
        for task in ("first", "second"):
            _write_list_files(data_root / "vtab-1k" / task)
        fast_run = "first"
        slow_run = "second"
    elif script_name == "tune_structured_testbest_stage1.sh":
        for task in ("dsprites_loc", "dsprites_ori", "clevr_dist"):
            _write_list_files(data_root / "vtab-1k" / task)
        fast_run = "dsprites_loc_direct_cat_gs_lr1e3"
        slow_run = "dsprites_ori_direct_cat_gs_lr1e3"
    else:
        raise AssertionError(script_name)

    bash_env = tmp_path / "bash-env"
    bash_env.write_text(
        """kill() {
  printf "%s\\n" "$*" >> "$KILL_EVENT_LOG"
  command kill "$@"
}
_wait_calls=0
wait() {
  local wait_rc
  if builtin wait "$@"; then wait_rc=0; else wait_rc=$?; fi
  _wait_calls=$((_wait_calls + 1))
  if (( _wait_calls == 1 )); then
    printf "reaped:%s\\n" "$1" >> "$WAIT_EVENT_LOG"
    while :; do command sleep 0.05; done
  fi
  return "$wait_rc"
}
"""
    )
    env = os.environ.copy()
    env.update(
        {
            "BASH_ENV": str(bash_env),
            "CHECKPOINT": str(checkpoint),
            "CLEVR_DATA_ROOT": str(data_root),
            "DATA_ROOT": str(data_root),
            "FAST_RUN": fast_run,
            "GPU_LIST": "0,1",
            "JOB_EVENT_LOG": str(tmp_path / "jobs.log"),
            "KILL_EVENT_LOG": str(tmp_path / "kills.log"),
            "LOG_ROOT": str(tmp_path / "logs"),
            "OUTPUT_ROOT": str(tmp_path / "outputs"),
            "PYTHON": str(fake_python),
            "WAIT_EVENT_LOG": str(tmp_path / "waits.log"),
        }
    )
    if script_name == "train_table1_vtab_daf.sh":
        env["TASKS"] = "first second"
    return env, fast_run, slow_run


@pytest.mark.parametrize(
    "script_name",
    (
        "reproduce_table1_vit_b16_npz_test_best.sh",
        "train_table1_vtab_daf.sh",
        "tune_structured_testbest_stage1.sh",
    ),
)
def test_signal_cleanup_never_kills_a_reaped_job_pid(tmp_path, script_name):
    env, fast_run, slow_run = _launcher_environment(tmp_path, script_name)
    script = ROOT / "scripts" / script_name
    process = subprocess.Popen(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    job_log = Path(env["JOB_EVENT_LOG"])
    wait_log = Path(env["WAIT_EVENT_LOG"])
    try:
        _wait_for(
            job_log,
            lambda: job_log.exists() and f"start:{slow_run}:" in job_log.read_text(),
        )
        _wait_for(
            wait_log,
            lambda: wait_log.exists() and wait_log.read_text().startswith("reaped:"),
        )
        events = job_log.read_text().splitlines()
        fast_pid = int(wait_log.read_text().strip().split(":", 1)[1])
        assert any(line == f"finish:{fast_run}:{fast_pid}" for line in events)
        slow_pid = int(next(line for line in events if line.startswith(f"start:{slow_run}:")).rsplit(":", 1)[1])

        process.terminate()
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    kill_log = Path(env["KILL_EVENT_LOG"])
    killed_pids = {
        int(line.split()[-1]) for line in kill_log.read_text().splitlines() if line.strip()
    }
    assert process.returncode != 0, (stdout, stderr)
    assert fast_pid not in killed_pids, "cleanup targeted a PID after wait had already reaped it"
    assert slow_pid in killed_pids, "cleanup did not terminate the still-active direct child"
    assert any(line == f"terminate:{slow_run}:{slow_pid}" for line in job_log.read_text().splitlines())


@pytest.mark.parametrize(
    "script_name",
    (
        "reproduce_table1_vit_b16_npz_test_best.sh",
        "train_table1_vtab_daf.sh",
        "tune_structured_testbest_stage1.sh",
    ),
)
def test_signal_cleanup_terminates_stopped_children_and_releases_locks(tmp_path, script_name):
    env, _fast_run, _slow_run = _launcher_environment(tmp_path, script_name)
    env["FAST_RUN"] = "no-run-finishes"
    Path(env["BASH_ENV"]).write_text(
        'kill() { printf "%s\\n" "$*" >> "$KILL_EVENT_LOG"; command kill "$@"; }\n'
    )
    script = ROOT / "scripts" / script_name
    process = subprocess.Popen(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    job_log = Path(env["JOB_EVENT_LOG"])
    started_pids = []
    timed_out = False
    stdout = ""
    stderr = ""
    elapsed = 0.0
    alive_after_launcher = set()
    locks_after_launcher = []
    try:
        _wait_for(
            job_log,
            lambda: job_log.exists()
            and len([line for line in job_log.read_text().splitlines() if line.startswith("start:")]) >= 2,
        )
        started_pids = [
            int(line.rsplit(":", 1)[1])
            for line in job_log.read_text().splitlines()
            if line.startswith("start:")
        ]
        stopped_pid = started_pids[0]
        os.kill(stopped_pid, signal.SIGSTOP)
        status_path = Path(f"/proc/{stopped_pid}/status")
        _wait_for(status_path, lambda: _pid_is_stopped(stopped_pid))

        started_at = time.monotonic()
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            timed_out = True
        elapsed = time.monotonic() - started_at
        alive_after_launcher = {pid for pid in started_pids if _pid_exists(pid)}
        locks_after_launcher = [
            root / ".daf-launch.lock"
            for root in (Path(env["OUTPUT_ROOT"]), Path(env["LOG_ROOT"]))
            if (root / ".daf-launch.lock").exists()
        ]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for pid in started_pids:
            if _pid_exists(pid):
                os.kill(pid, signal.SIGCONT)
                os.kill(pid, signal.SIGTERM)
        cleanup_deadline = time.monotonic() + 1
        while any(_pid_exists(pid) for pid in started_pids) and time.monotonic() < cleanup_deadline:
            time.sleep(0.01)
        for pid in started_pids:
            if _pid_exists(pid):
                os.kill(pid, signal.SIGKILL)

    assert not timed_out, "launcher cleanup hung while waiting for a stopped child"
    assert elapsed < 5
    assert process.returncode != 0, (stdout, stderr)
    assert not alive_after_launcher, f"launcher left managed children running: {alive_after_launcher}"
    assert not locks_after_launcher, f"launcher left locks behind: {locks_after_launcher}"


@pytest.mark.parametrize("tasks_value", (None, ""))
def test_auxiliary_vtab_launcher_requires_explicit_nonempty_tasks(tmp_path, tasks_value):
    invocation_log = tmp_path / "python-called"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(f"#!/usr/bin/env bash\ntouch {invocation_log!s}\n")
    fake_python.chmod(0o755)
    checkpoint = tmp_path / "weights.npz"
    checkpoint.touch()
    env = os.environ.copy()
    env.update({"CHECKPOINT": str(checkpoint), "PYTHON": str(fake_python)})
    if tasks_value is None:
        env.pop("TASKS", None)
    else:
        env["TASKS"] = tasks_value

    script = ROOT / "scripts" / "train_table1_vtab_daf.sh"
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
    assert "TASKS must be set" in completed.stderr
    assert not invocation_log.exists(), "missing TASKS must not launch training or summary Python"


def test_auxiliary_vtab_launcher_help_requires_tasks():
    script = ROOT / "scripts" / "train_table1_vtab_daf.sh"
    completed = subprocess.run(
        ["bash", str(script), "--help"], text=True, capture_output=True, timeout=10, check=False
    )

    assert completed.returncode == 0
    assert "TASKS must be set" in completed.stdout
