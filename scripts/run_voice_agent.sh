#!/usr/bin/env bash
# Start the Mobipick voice agent on the host (GUI button "Voice Agent").
# Needs: GPT Robot Demo (rosbridge :9090, the agents) and LiteLLM (:4000) running,
# and scripts/install_host.sh run once. Extra arguments go to realtime_voice_agent,
# e.g. --voice echo --input ALU1 --output Pebble. Environment overrides:
# REALTIME_VOICE, REALTIME_MODEL, REALTIME_INPUT_DEVICE, REALTIME_OUTPUT_DEVICE,
# LITELLM_BASE_URL, ROSBRIDGE_URL.
set -euo pipefail

# The GUI appends its toolbar dropdown as voice_agent:=off|cedar|echo.
# off keeps the classic workflow: commands come from the GUI or /recognized_speech.
args=()
for arg in "$@"; do
  case "$arg" in
    voice_agent:=off)
      printf 'voice agent is off (toolbar voice_agent=off): send commands through the GUI or /recognized_speech\n'
      exit 0 ;;
    voice_agent:=*) args+=(--voice "${arg#voice_agent:=}") ;;
    *) args+=("$arg") ;;
  esac
done
set -- "${args[@]+"${args[@]}"}"

package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_src="$(dirname -- "$package_dir")"
python="$package_dir/.venv/bin/python"
if [[ ! -x "$python" ]]; then
  printf 'no venv yet: run %s/scripts/install_host.sh once\n' "$package_dir" >&2
  exit 1
fi
if [[ -z "${LITELLM_API_KEY:-}" && -f "$workspace_src/litellm_master_key" ]]; then
  IFS= read -r LITELLM_API_KEY < "$workspace_src/litellm_master_key" || true
  export LITELLM_API_KEY
fi
export LITELLM_BASE_URL="${LITELLM_BASE_URL:-http://127.0.0.1:4000/v1}"
# The Mobipick containers live on the "mobipick" Docker network, so rosbridge
# (started by GPT Robot Demo) is not on the host's localhost. Find the
# container that accepts connections on :9090 (read-only docker inspection).
find_rosbridge() {
  local network="${MOBIPICK_DOCKER_NETWORK:-mobipick}" ip
  for ip in $(docker network inspect "$network" \
      -f '{{range .Containers}}{{.IPv4Address}} {{end}}' 2>/dev/null); do
    ip="${ip%/*}"
    if timeout 1 bash -c "echo > /dev/tcp/$ip/9090" 2>/dev/null; then
      printf 'ws://%s:9090' "$ip"
      return 0
    fi
  done
  printf 'ws://localhost:9090'
}
if [[ -z "${ROSBRIDGE_URL:-}" ]]; then
  # GPT Robot Demo may still be starting: wait up to 30 s for its rosbridge
  for _ in $(seq 1 30); do
    ROSBRIDGE_URL="$(find_rosbridge)"
    [[ "$ROSBRIDGE_URL" != "ws://localhost:9090" ]] && break
    sleep 1
  done
fi
export ROSBRIDGE_URL
printf 'rosbridge: %s\n' "$ROSBRIDGE_URL"
export PYTHONUNBUFFERED=1
exec "$python" "$package_dir/scripts/realtime_voice_agent" "$@"
