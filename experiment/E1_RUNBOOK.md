# E1 main run: phone runbook

4,200 calls at about 35 s each, so roughly 41 hours of continuous running. Keep the phone plugged in while it runs.
Being plugged in from the start is not required, because the run resumes after any interruption.

## Once, before the first start
1. Android Settings → Apps → Termux → Battery: **Unrestricted** (no battery optimisation).
2. File the OSF pre-registration. Then put its URL in `OSF_REGISTRATION.txt` in this folder. `resume_e1.sh` refuses to start without it.

## Start, or resume after a pause, power loss or reboot
Open Termux, then run:

    proot-distro login ubuntu
    bash /sdcard/Download/TIMONA_Export/TIMONA_Ideas/analysis_a/e1/resume_e1.sh start

The script takes the Termux wake lock, starts llama-server if it is down, and continues from the last saved call.
You can close the Termux screen afterwards, but do not swipe the Termux notification away.

## Check or stop

    bash .../resume_e1.sh status   # e.g. "progress: 1234/4200 calls"
    bash .../resume_e1.sh stop     # safe at any time; resume later with start

## What happens on failures
- **Power loss or reboot:** at most the one call in progress is lost. On restart, a half-written last line is removed and that call is re-run with the same seed. Nothing restarts by itself after a reboot, so run `start` again.
- **Server crash:** the script restarts llama-server and the run after 30 s. It gives up after 5 attempts in a row without progress (see `main_stdout.log`).
- **Heat:** the phone throttles, which slows the run but does not affect the results. Call order is randomised, so slowdowns do not line up with arms.
