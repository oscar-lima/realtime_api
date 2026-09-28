#!/usr/bin/env bash
# Route the voice agent's audio from the laptop to the real robot's USB speaker (issues #68, #102): a PulseAudio null
# sink "mobipick_robot" on the laptop, its monitor streamed over ssh into the robot's PulseAudio with pacat, and every
# 2 s the voice agent's playback streams (PipeWire ALSA [python3...]) moved to that sink. Restarts the ssh pipe when it
# dies. SIGINT/SIGTERM (the GUI's stop, or Ctrl-C) unloads the null sink again, which sends the voice back to the
# laptop's default output.
# The Mobipick Labs GUI's "Robot Speaker" button starts it; its option rule appends route:=on only with "Use remote ROS
# master" (the real robot). Without route:=on it exits at once, so Auto Launch in the simulation starts it harmlessly.
# Arguments (name:=value, like the voice agent's): route:=on|off  robot:=user@host (default robot@mobipick-os-sensor)
#   match:=<application.name substring> (default "PipeWire ALSA [python3")
ROUTE=off
ROBOT=robot@mobipick-os-sensor
MATCH='PipeWire ALSA [python3'
for arg in "$@"; do
  case "$arg" in
    route:=*) ROUTE="${arg#route:=}" ;;
    robot:=*) ROBOT="${arg#robot:=}" ;;
    match:=*) MATCH="${arg#match:=}" ;;
    *) printf 'unknown argument %s (route:=on|off robot:=user@host match:=text)\n' "$arg" >&2; exit 2 ;;
  esac
done
if [ "$ROUTE" != on ]; then
  printf 'robot speaker routing is off (route:=%s): the voice stays on the laptop; the GUI routes it only on the real robot\n' "$ROUTE"
  exit 0
fi
SINK=mobipick_robot
FMT=(--format=s16le --rate=24000 --channels=1)

mod=$(pactl list short modules | awk -v s="sink_name=$SINK" '$0 ~ s {print $1}')
if [ -z "$mod" ]; then
  mod=$(pactl load-module module-null-sink sink_name=$SINK sink_properties=device.description=$SINK)
  echo "loaded null sink $SINK (module $mod)"
fi
pipe_pid=
cleanup() {
  if [ -n "$pipe_pid" ]; then
    pkill -P "$pipe_pid" 2>/dev/null   # parec and ssh of the pipe subshell
    kill "$pipe_pid" 2>/dev/null
  fi
  pactl unload-module "$mod" 2>/dev/null
  echo "robot speaker routing stopped (null sink $SINK unloaded, voice back on the laptop)"
  exit 0
}
trap cleanup INT TERM HUP

start_pipe() {
  ( parec -d $SINK.monitor "${FMT[@]}" --latency-msec=40 \
      | ssh -o BatchMode=yes -o ServerAliveInterval=5 "$ROBOT" "pacat ${FMT[*]} --latency-msec=80" ) &
  pipe_pid=$!
  echo "$(date +%T) audio pipe to $ROBOT started (pid $pipe_pid)"
}
start_pipe
while true; do
  kill -0 $pipe_pid 2>/dev/null || { echo "$(date +%T) audio pipe died, restarting"; start_pipe; }
  sink_idx=$(pactl list short sinks | awk -v s=$SINK '$2==s {print $1}')
  pactl list sink-inputs | awk -v m="$MATCH" '
    /^Sink Input #/ {id=substr($3,2)} /^\s*Sink: / {sink=$2}
    /application.name = / && index($0, m) {print id, sink}' |
  while read -r id sink; do
    if [ "$sink" != "$sink_idx" ]; then pactl move-sink-input "$id" $SINK && echo "$(date +%T) moved stream $id to $SINK"; fi
  done
  sleep 2 &
  wait $!   # interruptible, so a stop is handled at once
done
