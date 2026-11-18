"""命令行入口的行为。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from facerec.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_help_exits_with_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


@pytest.mark.parametrize("command", ["collect", "train", "recognize", "pipeline", "doctor", "info"])
def test_subcommand_help_exits_with_zero(command):
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0


def test_info_on_empty_dataset(dataset_root, capsys):
    assert main(["info", "--data", str(dataset_root)]) == 0
    assert "还没有任何样本" in capsys.readouterr().out


def test_info_prints_table(dataset_root, make_files, capsys):
    make_files(dataset_root / "train" / "张三", 4)
    make_files(dataset_root / "val" / "张三", 1)

    assert main(["info", "--data", str(dataset_root)]) == 0
    output = capsys.readouterr().out
    assert "张三" in output
    assert "共 1 个类别，5 张图片" in output


def test_doctor_reports_a_verdict(dataset_root, capsys):
    code = main(["doctor", "--data", str(dataset_root)])
    assert code in (0, 1)
    output = capsys.readouterr().out
    assert "结论" in output
    assert "Python 版本" in output


def test_recognize_without_weights_fails_cleanly(tmp_path, capsys):
    code = main(["recognize", "--weights", str(tmp_path / "missing.pt")])
    assert code == 1


def test_main_py_runs_as_a_script():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0
    assert "collect" in result.stdout
