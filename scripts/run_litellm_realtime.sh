#!/usr/bin/env bash
# Start a LiteLLM proxy that serves the realtime aliases in
# config/litellm_realtime.yaml. Uses the same pinned image, key files and
# master key as mobipick_gpt/mobipick_labs_compatible_docker/run_litellm.sh.
#
#   scripts/run_litellm_realtime.sh            # port 4001 (4000 stays free for the main proxy)
#   LITELLM_PORT=4000 scripts/run_litellm_realtime.sh
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_dir="$(cd -- "$script_dir/.." && pwd)"
workspace_src="$(dirname -- "$package_dir")"
config_file="${LITELLM_CONFIG_FILE:-$package_dir/config/litellm_realtime.yaml}"
port="${LITELLM_PORT:-4001}"
image_env="$workspace_src/mobipick_gpt/mobipick_labs_compatible_docker/litellm_image.env"
image="docker.litellm.ai/berriai/litellm:v1.102.1"
if [[ -f "$image_env" ]]; then
  # shellcheck disable=SC1090
  source "$image_env"
  image="${LITELLM_IMAGE:-$LITELLM_DEFAULT_IMAGE}"
fi

read_key() {
  local key=""
  if [[ -f "$1" ]]; then
    IFS= read -r key < "$1" || true  # tolerate a missing trailing newline
  fi
  printf '%s' "$key"
}
export OPENAI_API_KEY="${OPENAI_API_KEY:-$(read_key "${OPENAI_KEY_FILE:-$workspace_src/openai_api_key}")}"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-$(read_key "${LITELLM_MASTER_KEY_FILE:-$workspace_src/litellm_master_key}")}"
: "${OPENAI_API_KEY:?no OpenAI key (set OPENAI_API_KEY or create $workspace_src/openai_api_key)}"
: "${LITELLM_MASTER_KEY:?no LiteLLM master key (run mobipick_gpt install.sh once or set LITELLM_MASTER_KEY)}"

name="litellm-realtime-$port"
docker container rm --force "$name" >/dev/null 2>&1 || true
printf 'LiteLLM realtime proxy on ws://127.0.0.1:%s/v1/realtime (config %s)\n' "$port" "$config_file"
exec docker run --rm --name "$name" \
  --publish "$port:4000" \
  --env OPENAI_API_KEY \
  --env LITELLM_MASTER_KEY \
  --volume "$config_file:/app/config.yaml:ro" \
  "$image" \
  --config /app/config.yaml --port 4000
