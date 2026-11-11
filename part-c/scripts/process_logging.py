#!/usr/bin/env python3
"""Persist child output during execution; retain diagnostics and stop MPI on timeout."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def run_logged(command, stem, *, env=None, timeout=90):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    stdout_path = Path(str(stem) + '.stdout')
    stderr_path = Path(str(stem) + '.stderr')
    status_path = Path(str(stem) + '.process.json')
    Path(str(stem) + '.command.json').write_text(json.dumps(command, indent=2) + '\n')
    record = dict(status='STARTING', timeout_s=timeout, command=command)
    def save():
        status_path.write_text(json.dumps(record, indent=2) + '\n')
    save()
    started = time.monotonic()
    timed_out = False
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        try:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        except OSError as exc:
            record.update(status='START_FAILED', error=str(exc))
            save()
            raise
        record.update(status='RUNNING', pid=process.pid)
        save()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Give mpirun time to terminate its ranks before the PBS job exits.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    record.update(status='TIMEOUT' if timed_out else ('COMPLETE' if process.returncode == 0 else 'FAILED'),
                  elapsed_s=time.monotonic()-started, returncode=process.returncode)
    save()
    output = stdout_path.read_text(errors='replace')
    errors = stderr_path.read_text(errors='replace')
    if timed_out or process.returncode:
        detail = '\n'.join((output[-4000:], errors[-8000:])).strip()
        reason = f'timed out after {timeout}s' if timed_out else f'exit={process.returncode}'
        raise RuntimeError(f'{stem.name}: {reason}; logs retained at {stem}\n{detail}')
    return subprocess.CompletedProcess(command, process.returncode, output, errors)
