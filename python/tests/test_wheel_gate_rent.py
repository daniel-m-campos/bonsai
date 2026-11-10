"""Tests for scripts/wheel_gate_rent.sh, the step that rents the wheel gate's pod.

The script runs with a fake ``curl`` and ``sleep`` first on ``PATH``. The fake curl answers
each pod create from a scripted list, reports a public SSH endpoint only for the pods a test
names, and logs every create, delete and wait, so a test reads which card each create named
and what the script did between them.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
from dataclasses import dataclass

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RENT_SCRIPT = REPO_ROOT / "scripts" / "wheel_gate_rent.sh"
IMAGE = "ghcr.io/example/bonsai:candidate-v0.0.0"
PUBLIC_KEY = "ssh-ed25519 AAAAFAKE gate"
L40S = "NVIDIA L40S"
RTX_4090 = "NVIDIA GeForce RTX 4090"
RTX_5090 = "NVIDIA GeForce RTX 5090"
CARDS = [L40S, RTX_4090, RTX_5090]
REFUSED = "refused"
NULL_ID = "nullid"
BACKOFF_SECONDS = "150"
POLL_SECONDS = "10"
POLLS_PER_POD = 36
SSH_ENDPOINT = ["ip=203.0.113.7", "port=40022"]

FAKE_CURL = r"""#!/bin/bash
args="$(echo "$*" | tr '\n' ' ')"
case "$args" in
  *"-X DELETE"*)
    echo "${args##*/v1/pods/}" | cut -d' ' -f1 >> "$FAKE_DIR/deletes"
    exit 0 ;;
  *graphql*)
    pod="$(echo "$args" | sed -n 's/.*podId:\\"\([^\\]*\)\\".*/\1/p')"
    if grep -qx "$pod" "$FAKE_DIR/pods_with_ssh"; then
      cat "$FAKE_DIR/runtime_with_ssh.json"
    else
      echo '{"data":{"pod":{"runtime":null}}}'
    fi
    exit 0 ;;
  *rest.runpod.io/v1/pods*)
    echo "$args" >> "$FAKE_DIR/creates"
    answer="$(sed -n "$(wc -l < "$FAKE_DIR/creates" | tr -d ' ')p" "$FAKE_DIR/create_answers")"
    case "$answer" in
      ok:*) echo "{\"id\": \"${answer#ok:}\"}"; exit 0 ;;
      nullid) echo '{"id": null}'; exit 0 ;;
      *) exit 22 ;;
    esac ;;
esac
echo "unexpected curl: $args" >&2
exit 99
"""
FAKE_SLEEP = """#!/bin/bash
echo "$1" >> "$FAKE_DIR/sleeps"
"""

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("jq") is None,
    reason="scripts/wheel_gate_rent.sh needs bash and jq on PATH",
)


@dataclass
class Rental:
    """What one run of the rent script did and reported."""

    exit_code: int
    stdout: str
    creates: list[str]
    deleted_pods: list[str]
    waits: list[str]
    outputs: list[str]

    @property
    def cards(self) -> list[str]:
        """The card each create named, in order."""
        return [_card_named(create) for create in self.creates]


def test_wheel_gate_rent_writes_the_pod_and_its_public_ssh_endpoint(tmp_path):
    rental = _rent(tmp_path, create_answers=["ok:podA"], pods_with_ssh=["podA"])
    assert rental.exit_code == 0
    assert rental.outputs == ["pod_id=podA", *SSH_ENDPOINT]
    assert rental.cards == [L40S]
    assert rental.waits == [POLL_SECONDS]
    assert rental.deleted_pods == []


def test_wheel_gate_rent_asks_for_the_image_under_the_gate_pod_name(tmp_path):
    rental = _rent(tmp_path, create_answers=["ok:podA"], pods_with_ssh=["podA"])
    (create,) = rental.creates
    assert f'"imageName": "{IMAGE}"' in create
    assert '"name": "bonsai-wheel-gate-test"' in create
    assert f'"PUBLIC_KEY": "{PUBLIC_KEY}"' in create
    assert '"cloudType": "SECURE"' in create
    assert "Authorization: Bearer fake-key" in create


def test_wheel_gate_rent_asks_for_the_next_card_when_one_is_refused(tmp_path):
    rental = _rent(tmp_path, create_answers=[REFUSED, "ok:podB"], pods_with_ssh=["podB"])
    assert rental.exit_code == 0
    assert rental.cards == [L40S, RTX_4090]
    assert rental.waits == [POLL_SECONDS]
    assert rental.outputs == ["pod_id=podB", *SSH_ENDPOINT]


def test_wheel_gate_rent_reads_a_null_pod_id_as_a_refused_create(tmp_path):
    answers = [NULL_ID, REFUSED, "ok:podC"]
    rental = _rent(tmp_path, create_answers=answers, pods_with_ssh=["podC"])
    assert rental.exit_code == 0
    assert rental.cards == CARDS
    assert rental.waits == [POLL_SECONDS]
    assert rental.outputs == ["pod_id=podC", *SSH_ENDPOINT]


def test_wheel_gate_rent_backs_off_only_after_every_card_is_refused(tmp_path):
    answers = [REFUSED, REFUSED, REFUSED, "ok:podB"]
    rental = _rent(tmp_path, create_answers=answers, pods_with_ssh=["podB"])
    assert rental.exit_code == 0
    assert rental.cards == [*CARDS, L40S]
    assert rental.waits == [BACKOFF_SECONDS, POLL_SECONDS]
    assert rental.outputs == ["pod_id=podB", *SSH_ENDPOINT]


def test_wheel_gate_rent_fails_after_five_rounds_of_refused_cards(tmp_path):
    rental = _rent(tmp_path, create_answers=[], pods_with_ssh=[])
    assert rental.exit_code == 1
    assert rental.cards == CARDS * 5
    assert rental.waits == [BACKOFF_SECONDS] * 5
    assert rental.outputs == []
    assert rental.deleted_pods == []
    assert "::error::no usable pod after 5 attempts" in rental.stdout


def test_wheel_gate_rent_deletes_a_pod_with_no_ssh_endpoint_and_rents_again(tmp_path):
    answers = ["ok:podD", REFUSED, "ok:podE"]
    rental = _rent(tmp_path, create_answers=answers, pods_with_ssh=["podE"])
    assert rental.exit_code == 0
    assert rental.deleted_pods == ["podD"]
    assert rental.cards == [L40S, L40S, RTX_4090]
    assert rental.waits == [POLL_SECONDS] * (POLLS_PER_POD + 1)
    assert rental.outputs == ["pod_id=podD", "pod_id=podE", *SSH_ENDPOINT]


def test_wheel_gate_rent_fails_after_five_pods_without_an_ssh_endpoint(tmp_path):
    pods = ["p1", "p2", "p3", "p4", "p5"]
    rental = _rent(tmp_path, create_answers=[f"ok:{pod}" for pod in pods], pods_with_ssh=[])
    assert rental.exit_code == 1
    assert rental.deleted_pods == pods
    assert rental.cards == [L40S] * 5


def test_wheel_gate_rent_refuses_to_run_without_the_image(tmp_path):
    rental = _rent(tmp_path, create_answers=["ok:podA"], pods_with_ssh=["podA"], unset=["IMG"])
    assert rental.exit_code == 2
    assert rental.creates == []
    assert "IMG is unset" in rental.stdout


# Private Functions ================================================================================


def _rent(
    tmp_path: pathlib.Path,
    create_answers: list[str],
    pods_with_ssh: list[str],
    unset: list[str] | None = None,
) -> Rental:
    """Run the rent script against the fake curl, one scripted answer per create."""
    fake_dir = tmp_path / "fake"
    fake_bin = tmp_path / "bin"
    fake_dir.mkdir()
    fake_bin.mkdir()
    _write_executable(fake_bin / "curl", FAKE_CURL)
    _write_executable(fake_bin / "sleep", FAKE_SLEEP)
    _write_lines(fake_dir / "create_answers", create_answers)
    _write_lines(fake_dir / "pods_with_ssh", pods_with_ssh)
    (fake_dir / "runtime_with_ssh.json").write_text(json.dumps(_runtime_with_ssh()))
    public_key_file = tmp_path / "podkey.pub"
    public_key_file.write_text(PUBLIC_KEY + "\n")
    outputs = tmp_path / "github_output"
    outputs.touch()
    env = {
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_DIR": str(fake_dir),
        "RUNPOD_KEY": "fake-key",
        "POD_NAME": "bonsai-wheel-gate-test",
        "IMG": IMAGE,
        "POD_PUBLIC_KEY_FILE": str(public_key_file),
        "GITHUB_OUTPUT": str(outputs),
    }
    for name in unset or []:
        del env[name]
    done = subprocess.run(
        ["bash", str(RENT_SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    return Rental(
        exit_code=done.returncode,
        stdout=done.stdout,
        creates=_read_lines(fake_dir / "creates"),
        deleted_pods=_read_lines(fake_dir / "deletes"),
        waits=_read_lines(fake_dir / "sleeps"),
        outputs=_read_lines(outputs),
    )


def _card_named(create: str) -> str:
    named = re.search(r'"gpuTypeIds": \["([^"]*)"\]', create)
    assert named is not None, create
    return named.group(1)


def _runtime_with_ssh() -> dict:
    """The runtime reply for a live pod: a private mapping and another port beside SSH."""
    ports = [
        {"ip": "100.65.0.1", "isIpPublic": False, "privatePort": 22, "publicPort": 60001},
        {"ip": "203.0.113.7", "isIpPublic": True, "privatePort": 8888, "publicPort": 40088},
        {"ip": "203.0.113.7", "isIpPublic": True, "privatePort": 22, "publicPort": 40022},
    ]
    return {"data": {"pod": {"runtime": {"uptimeInSeconds": 5, "ports": ports}}}}


def _write_executable(path: pathlib.Path, body: str):
    path.write_text(body)
    path.chmod(0o755)


def _write_lines(path: pathlib.Path, lines: list[str]):
    path.write_text("".join(f"{line}\n" for line in lines))


def _read_lines(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text().splitlines()
