#!/usr/bin/env bash
# E1 main run controller (E1_OSF_PREREG.md). Starts or resumes run_main.py in the background, restarts llama-server
# and the run after a crash, and stops after 5 attempts in a row without progress.
#   bash resume_e1.sh start    start, or resume after a pause, power loss or reboot
#   bash resume_e1.sh status   progress, runner and server state, last log lines
#   bash resume_e1.sh stop     stop the runner and the server (safe at any time; the run resumes where it stopped)
# start refuses to run until OSF_REGISTRATION.txt (in this folder) holds the OSF registration URL.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SERVER_BIN=/root/llama.cpp/build/bin/llama-server
MODEL=/root/models/gemma-4-E2B-it-Q4_0.gguf
REG="$HERE/OSF_REGISTRATION.txt"
LOG="$HERE/main_log.jsonl"
OUT="$HERE/main_stdout.log"
PID="$HERE/.e1_runner.pid"
TOTAL=4200

server_up() { curl -sf http://127.0.0.1:8089/health >/dev/null; }
runner_alive() { [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; }
done_count() { if [ -f "$LOG" ]; then wc -l < "$LOG"; else echo 0; fi; }
stop_server() { pgrep -x llama-server | xargs -r kill; }   # never pkill -f: it matches and kills the calling shell

start_server() {
  server_up && return 0
  stop_server; sleep 2
  nohup "$SERVER_BIN" -m "$MODEL" -t 6 -c 4096 --host 127.0.0.1 --port 8089 --no-webui >> /root/llama_server.log 2>&1 &
  for _ in $(seq 1 120); do server_up && return 0; sleep 2; done
  return 1
}

loop() {
  local fails=0 before
  while :; do
    before=$(done_count)
    if start_server; then
      if python3 "$HERE/run_main.py" --i-have-registered; then
        echo "$(date '+%F %T') E1 complete: $(done_count)/$TOTAL records"; break
      fi
    else
      echo "$(date '+%F %T') llama-server did not come up within 4 minutes"
    fi
    if [ "$(done_count)" -gt "$before" ]; then fails=1; else fails=$((fails + 1)); fi
    if [ "$fails" -ge 5 ]; then echo "$(date '+%F %T') stopping: 5 attempts in a row without progress"; exit 1; fi
    echo "$(date '+%F %T') run stopped at $(done_count)/$TOTAL; restarting server and run in 30 s"
    stop_server; sleep 30
  done
  stop_server
  command -v termux-wake-unlock >/dev/null && termux-wake-unlock
  rm -f "$PID"
}

case "${1:-status}" in
  start)
    [ -s "$REG" ] || { echo "Refusing to start: put the OSF registration URL in $REG first."; exit 1; }
    runner_alive && { echo "Already running (pid $(cat "$PID")): $(done_count)/$TOTAL"; exit 0; }
    command -v termux-wake-lock >/dev/null && termux-wake-lock
    echo "$(date '+%F %T') start/resume at $(done_count)/$TOTAL" >> "$OUT"
    setsid nohup bash "$0" _loop >> "$OUT" 2>&1 < /dev/null &
    echo $! > "$PID"
    echo "Started (pid $!). Progress: bash $0 status"
    ;;
  _loop) loop ;;
  status)
    echo "progress: $(done_count)/$TOTAL calls"
    if runner_alive; then echo "runner: running (pid $(cat "$PID"))"; else echo "runner: not running"; fi
    if server_up; then echo "server: up"; else echo "server: down"; fi
    [ -f "$OUT" ] && tail -n 3 "$OUT"
    ;;
  stop)
    runner_alive && kill -- -"$(cat "$PID")" 2>/dev/null
    rm -f "$PID"; stop_server
    command -v termux-wake-unlock >/dev/null && termux-wake-unlock
    echo "Stopped at $(done_count)/$TOTAL. Resume with: bash $0 start"
    ;;
  *) echo "usage: bash $0 start|status|stop"; exit 2 ;;
esac
