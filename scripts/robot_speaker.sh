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
#   clean:=on|off (default off, or env ROBOT_SPEAKER_CLEAN=1): the robot-side pacat is tagged (client name
#   mobipick_robot_speaker); before each pipe start and on stop, this script's earlier ones are killed on the robot, and
#   3 s after a start it checks that exactly one plays. #187: after a network drop the old pacat kept running next to
#   the new one (its ssh was gone, the robot never got EOF) and the robot stayed silent until `pkill -x pacat`. Off
#   keeps the behaviour of the SAB demo (2026-09-28) until clean:=on is tested on the robot.
ROUTE=off
ROBOT=robot@mobipick-os-sensor
MATCH='PipeWire ALSA [python3'
CLEAN=off
[ "${ROBOT_SPEAKER_CLEAN:-0}" = 1 ] && CLEAN=on
for arg in "$@"; do
  case "$arg" in
    route:=*) ROUTE="${arg#route:=}" ;;
    robot:=*) ROBOT="${arg#robot:=}" ;;
    match:=*) MATCH="${arg#match:=}" ;;
    clean:=*) CLEAN="${arg#clean:=}" ;;
    *) printf 'unknown argument %s (route:=on|off robot:=user@host match:=text clean:=on|off)\n' "$arg" >&2; exit 2 ;;
  esac
done
if [ "$ROUTE" != on ]; then
  printf 'robot speaker routing is off (route:=%s): the voice stays on the laptop; the GUI routes it only on the real robot\n' "$ROUTE"
  exit 0
fi
SINK=mobipick_robot
FMT=(--format=s16le --rate=24000 --channels=1)
REMOTE="pacat ${FMT[*]} --latency-msec=80"
TAG=mobipick_robot_speaker
OURS="^pacat .*--client-name=$TAG"  # our pacat on the robot; the ssh shell and pgrep/pkill do not start with pacat
if [ "$CLEAN" = on ]; then
  REMOTE="pkill -f '$OURS'; exec $REMOTE --client-name=$TAG"
  echo "clean:=on: earlier robot-side pacat of this script are killed before each start and on stop"
fi
robot() { timeout 10 ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROBOT" "$@"; }

mod=$(pactl list short modules | awk -v s="sink_name=$SINK" '$0 ~ s {print $1}')
if [ -z "$mod" ]; then
  mod=$(pactl load-module module-null-sink sink_name=$SINK sink_properties=device.description=$SINK)
  echo "loaded null sink $SINK (module $mod)"
fi
pipe_pid=
check_at=
cleanup() {
  if [ -n "$pipe_pid" ]; then
    pkill -P "$pipe_pid" 2>/dev/null   # parec and ssh of the pipe subshell
    kill "$pipe_pid" 2>/dev/null
  fi
  pactl unload-module "$mod" 2>/dev/null
  if [ "$CLEAN" = on ]; then
    robot "pkill -f '$OURS'" 2>/dev/null   # a half-open ssh leaves the robot's pacat waiting for more audio
  fi
  echo "robot speaker routing stopped (null sink $SINK unloaded, voice back on the laptop)"
  exit 0
}
trap cleanup INT TERM HUP

start_pipe() {
  ( parec -d $SINK.monitor "${FMT[@]}" --latency-msec=40 \
      | ssh -o BatchMode=yes -o ServerAliveInterval=5 "$ROBOT" "$REMOTE" ) &
  pipe_pid=$!
  check_at=$((SECONDS + 3))
  echo "$(date +%T) audio pipe to $ROBOT started (pid $pipe_pid)"
}

check_robot() {  # clean:=on: exactly one pacat of ours plays on the robot
  local counts procs inputs
  counts=$(robot "echo \$(pgrep -fc '$OURS') \$(pactl list sink-inputs 2>/dev/null | grep -c 'application.name = \"$TAG\"')")
  read -r procs inputs <<< "$counts"
  if [ "$procs" = 1 ]; then
    echo "$(date +%T) robot plays one speaker stream (pacat 1, sink inputs ${inputs:-?})"
    return
  fi
  echo "$(date +%T) WARNING: robot has ${procs:-?} speaker pacat (sink inputs ${inputs:-?}), expected 1"
  if [ "${procs:-0}" -gt 1 ] 2>/dev/null; then
    robot "for i in 1 2 3 4 5; do [ \$(pgrep -fc '$OURS') -gt 1 ] || break; pkill -o -f '$OURS'; sleep 0.2; done"
    echo "$(date +%T) killed the older ones, keeping the newest"
  fi
}
start_pipe
while true; do
  kill -0 $pipe_pid 2>/dev/null || { echo "$(date +%T) audio pipe died, restarting"; start_pipe; }
  if [ "$CLEAN" = on ] && [ -n "$check_at" ] && [ "$SECONDS" -ge "$check_at" ]; then
    check_at=
    check_robot
  fi
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
