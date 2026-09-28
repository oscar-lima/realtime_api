"""run_voice_agent.sh finds rosbridge on localhost at once (#62: the real robot's demo uses host networking)."""

import os
import socket
import subprocess
import time

import pytest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_voice_agent.sh")
VENV_PYTHON = os.path.join(os.path.dirname(__file__), "..", ".venv", "bin", "python")
# the script checks its host venv first (scripts/install_host.sh); elsewhere (Noetic image) there is none
pytestmark = pytest.mark.skipif(not os.access(VENV_PYTHON, os.X_OK), reason="no runnable host venv")


def find(tmp_path, port, network_ips="", wait_s=30):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    (fake / "docker").write_text("#!/bin/sh\nprintf '%s'\n" % network_ips)
    (fake / "docker").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", ROSBRIDGE_PORT=str(port), ROSBRIDGE_ONLY_FIND="1",
               ROSBRIDGE_WAIT_S=str(wait_s))
    env.pop("ROSBRIDGE_URL", None)
    started = time.monotonic()
    out = subprocess.run(["bash", SCRIPT], env=env, capture_output=True, text=True, timeout=60)
    return out.stdout.strip().splitlines()[-1], time.monotonic() - started


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_rosbridge_on_localhost_is_taken_at_once(tmp_path):
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    port = server.getsockname()[1]
    try:
        url, took = find(tmp_path, port)
    finally:
        server.close()
    assert url == f"ws://localhost:{port}" and took < 5.0


def test_without_rosbridge_it_falls_back_to_localhost_after_the_wait(tmp_path):
    port = free_port()
    url, took = find(tmp_path, port, wait_s=2)
    assert url == f"ws://localhost:{port}" and took > 1.5   # tries once a second, then the old fallback
