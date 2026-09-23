#!/usr/bin/env bash
# One-time host setup for the voice agent: a venv inside this package with the
# audio, echo cancellation (WebRTC via livekit) and rosbridge dependencies.
#   scripts/install_host.sh
set -euo pipefail
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
missing=()
for lib in libportaudio.so.2 libspeexdsp.so.1; do
  ldconfig -p | grep "$lib" >/dev/null || missing+=("$lib")
done
if ((${#missing[@]})); then
  printf 'missing system libraries: %s\n  sudo apt install libportaudio2 libspeexdsp1\n' "${missing[*]}" >&2
fi
python3 -m venv "$package_dir/.venv"
"$package_dir/.venv/bin/pip" install --quiet --upgrade pip
"$package_dir/.venv/bin/pip" install --quiet -r "$package_dir/requirements.txt"
PYTHONPATH="$package_dir/src" "$package_dir/.venv/bin/python" -c \
  'from realtime_api.echo_cancel import available_backends; print("AEC backends:", available_backends())'
printf 'ready: %s/scripts/run_voice_agent.sh\n' "$package_dir"
