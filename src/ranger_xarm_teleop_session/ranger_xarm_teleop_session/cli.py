# Copyright 2026 Shuaib Olanrewaju
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
teleop_session: start, check and stop a teleoperation session with one command.

    tools/teleop_session start --profile gazebo        # everything on this machine
    tools/teleop_session start --profile mock
    tools/teleop_session start --role relay [--listen ADDRESS]   # operator PC, no ROS needed
    tools/teleop_session start --profile gazebo --relay ws://OPERATOR-PC:8443/ws
    tools/teleop_session stop                          # or Ctrl-C where it runs
    tools/teleop_session status

One session at a time per machine and user. The task label and operator are typed at the
prompt and written only into the data directory. Logs go to
``$ROS_LOG_DIR/teleop_session/<session>/`` (default ``~/.ros/log``).
"""

import argparse
import os
import secrets
import shutil
import signal
import sys
import time
from typing import List, Optional

from . import plan as pl
from . import preflight as pf
from . import procs
from . import supervisor as sv

SELF_VAR = 'TELEOP_SESSION_SELF'      # set on the re-exec that gives the supervisor its mark
STOP_SUPERVISOR_WAIT_S = 600.0
TRUE_DEFAULTS = ('record', 'feedback', 'reset', 'cameras', 'rviz')


def checkout_of(path: str) -> Optional[str]:
    """Return the checkout holding this package's source (where tools/relay.sh is), or None."""
    p = os.path.dirname(os.path.realpath(path))
    for _ in range(6):
        if os.path.isfile(os.path.join(p, 'tools', 'relay.sh')):
            return p
        p = os.path.dirname(p)
    env = os.environ.get('TELEOP_CHECKOUT')
    return env if env and os.path.isfile(os.path.join(env, 'tools', 'relay.sh')) else None


def parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(prog='teleop_session', description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='command', required=True)
    st = sub.add_parser('start', help='start a session (foreground; Ctrl-C stops it)')
    st.add_argument('--profile', default='mock', help=f'one of {", ".join(pl.PROFILES)}')
    st.add_argument('--role', default='stack', choices=pl.ROLES,
                    help='stack: the ROS stack (and the relay, unless --relay); relay: only '
                         'the relay, on the operator PC (no ROS needed)')
    for name, text in (('record', 'the recorder'), ('feedback', 'operator feedback'),
                       ('reset', 'the hold-to-run reset (simulation only)'),
                       ('cameras', "the simulator's cameras"), ('rviz', 'RViz (mock)')):
        st.add_argument(f'--no-{name}', dest=name, action='store_false', default=None,
                        help=f'switch off {text}')
    st.add_argument('--headless', action='store_true', default=None,
                    help='simulator without its window')
    st.add_argument('--gpu', default=None, choices=pl.GPU_CHOICES,
                    help='where the simulator renders (default auto: the NVIDIA card if any)')
    st.add_argument('--relay', dest='relay_url', default=None,
                    help='ws://HOST:PORT/ws of a relay on another machine (none starts here)')
    st.add_argument('--listen', default=None,
                    help=f'address the relay listens on (default {pl.DEFAULT_HOST}, this '
                         'machine only). The relay has no authentication')
    st.add_argument('--port', type=int, default=None)
    st.add_argument('--data-dir', default=None, help="default: the profile's data_dir")
    st.add_argument('--sim-check-seconds', type=float, default=None)
    st.add_argument('--keep-labels', action='store_true',
                    help='no prompt: use the labels already in the data directory')
    mode = st.add_mutually_exclusive_group()
    mode.add_argument('--yes', action='store_true', help='go on past missing items')
    mode.add_argument('--strict', action='store_true', help='stop at the first missing item')
    sub.add_parser('stop', help='stop the running session (also when its supervisor is gone)')
    sub.add_parser('status', help='say whether a session runs')
    return ap


def options_from(args) -> pl.Options:
    """Return the Options for the parsed ``start`` arguments, noting which were typed."""
    opts = pl.Options()
    explicit = []
    for name in ('profile', 'role', 'record', 'feedback', 'reset', 'headless', 'cameras',
                 'gpu', 'rviz', 'relay_url', 'listen', 'port', 'data_dir', 'sim_check_seconds'):
        value = getattr(args, name)
        if value is None:
            continue
        if name not in ('profile', 'role') or value != getattr(opts, name):
            explicit.append(name)
        setattr(opts, name, value)
    opts.explicit = tuple(explicit)
    return opts


def refuse_second_session(state_file: str, out=print) -> Optional[int]:
    """Refuse when a live session's state file is there; report and clear a stale one."""
    state = procs.read_state(state_file)
    if state is None:
        return None
    if state.get('unreadable'):
        out(f'an unreadable state file {state_file} was there; cleared (no session can be '
            'found from it)')
        procs.clear_state(state_file)
        return None
    if procs.session_is_live(state):
        out(f'refused: session {state.get("session")} (profile {state.get("profile")}, role '
            f'{state.get("role")}, phase {state.get("phase")}) is running on this machine. '
            'One session at a time: stop it first with  tools/teleop_session stop')
        return sv.EXIT_REFUSED
    out(f'a stale state file of session {state.get("session")} was there: no process carries '
        'its mark. Cleared, not obeyed')
    procs.clear_state(state_file)
    return None


def main(argv: Optional[List[str]] = None, checkout: Optional[str] = None) -> int:
    """Run a command."""
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    checkout = checkout or checkout_of(__file__)
    state_file = procs.state_path()
    if args.command == 'stop':
        return stop_command(state_file)
    if args.command == 'status':
        return status_command(state_file)
    return start_command(args, checkout, state_file, argv)


def start_command(args, checkout: Optional[str], state_file: str, argv) -> int:
    """Start a session (re-executes itself once to carry the session's mark)."""
    if not checkout:
        print('teleop_session: cannot find the checkout (tools/relay.sh); run '
              'tools/teleop_session from it')
        return sv.EXIT_REFUSED
    mark = os.environ.get(procs.MARK_VAR)
    if not mark or os.environ.get(SELF_VAR) != mark:
        refused = refuse_second_session(state_file)
        if refused is not None:
            return refused
        mark = pl.session_name() + '.' + secrets.token_hex(4)
        env = dict(os.environ, **{procs.MARK_VAR: mark, SELF_VAR: mark})
        script = os.path.realpath(sys.argv[0])
        if script.endswith(os.path.join('ranger_xarm_teleop_session', 'cli.py')):
            print('teleop_session: start it with tools/teleop_session (or the teleop_session '
                  'console script), not python -m')
            return sv.EXIT_REFUSED
        os.execve(sys.executable, [sys.executable, script] + list(
            sys.argv[1:] if argv is None else argv), env)
    session = mark.split('.')[0]
    opts = options_from(args)
    record_profile = None
    if opts.role == 'stack':
        problem = ros_problem()
        if problem:
            print(problem)
            return sv.EXIT_REFUSED
        try:
            record_profile = load_record_profile(opts.profile)
        except Exception as err:  # noqa: B902 - any unreadable profile is reported plainly
            print(f'refused: {err}')
            return sv.EXIT_REFUSED
    else:
        for name in ('record', 'feedback', 'reset'):
            setattr(opts, name, False)
    try:
        warnings = pl.check_options(opts, record_profile)
    except pl.PlanError as err:
        print(f'refused: {err}')
        return sv.EXIT_REFUSED
    log_dir = os.path.join(os.path.expanduser(os.environ.get('ROS_LOG_DIR')
                                              or os.path.join('~', '.ros', 'log')),
                           'teleop_session', session)
    os.makedirs(log_dir, exist_ok=True)
    console = sv.Console(log_path=os.path.join(log_dir, 'launcher.log'))
    console.say(f'session {session}; logs in {log_dir}')
    for w in warnings:
        console.say(f'note: {w}')
    relay_here = opts.role == 'relay' or not opts.relay_url
    if relay_here and not pl.is_loopback(opts.listen):
        console.say(f'the relay listens on {opts.listen}: it has no authentication, so anything '
                    'on that network can connect to it')
    if args.yes:
        policy = 'yes'
    elif args.strict or not sys.stdin.isatty():
        policy = 'strict'
    else:
        policy = 'ask'
    if policy == 'strict' and not args.strict:
        console.say('note: no terminal to ask in; a missing item stops the session '
                    '(--yes goes on)')
    sup = sv.Supervisor(opts, console, checkout=checkout, mark=mark, state_file=state_file,
                        session=session, log_dir=log_dir, policy=policy,
                        keep_labels=args.keep_labels)

    def on_signal(signum, _frame):
        if not sup.stopping:
            sup.stopping = True
            raise sv.StopRequested()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, on_signal)
    sup.started_wall = time.time()
    sup.write_state()
    try:
        local = local_items(opts, sup, record_profile, policy == 'ask' and not args.keep_labels)
        sim = sup_sim(opts)
    except sv.StopRequested:
        sup.say('stop asked for')
        sup.stop()
        return sv.EXIT_OK
    except sv.Refused as why:
        sup.say(str(why))
        sup.stop()
        return sv.EXIT_REFUSED
    return sup.run(local + ([sim['item']] if sim else []), record_profile, sim)


def ros_problem() -> Optional[str]:
    """Return why the stack role cannot run here, or None."""
    if not os.environ.get('AMENT_PREFIX_PATH') or shutil.which('ros2') is None:
        return ('refused: the stack role needs ROS sourced (the robot workspace and this '
                "checkout's install/setup.bash); the relay role does not")
    return None


def load_record_profile(name: str) -> dict:
    """Return the deployment profile of a launcher profile."""
    from ranger_xarm_teleop_record.profile import load_profile
    if name not in pl.PROFILES:
        return {}
    return load_profile(pl.PROFILES[name].record_profile)


def sup_sim(opts: pl.Options) -> Optional[dict]:
    """Return the simulator's settings and the GPU item (None without a simulator)."""
    if opts.role != 'stack' or pl.PROFILES[opts.profile].simulator is None:
        return None
    from .facts import nvidia_found
    found = nvidia_found()
    renders = opts.cameras or not opts.headless
    gpu, sentence = pl.gpu_choice(opts.gpu, found, renders)
    return {'gpu': gpu, 'headless': opts.headless, 'cameras': opts.cameras,
            'item': pf.gpu_item(True, gpu, sentence, found, renders)}


def local_items(opts: pl.Options, sup: 'sv.Supervisor', record_profile: Optional[dict],
                prompt: bool) -> List[pf.Item]:
    """Return the items judged before anything starts (asking for the labels first)."""
    from .facts import adb_facts, free_gb, robot_workspace
    parts = ({} if opts.role == 'relay' else
             {'recording': opts.record, 'feedback': opts.feedback, 'reset': opts.reset})
    sim = (None if opts.role == 'relay' or pl.PROFILES[opts.profile].simulator is None
           else {'headless': opts.headless, 'cameras': opts.cameras})
    items = [pf.plan_item(opts.profile if opts.role == 'stack' else '-', opts.role, parts, sim)]
    if opts.role == 'stack':
        from ranger_xarm_teleop_record import metadata as md
        from ranger_xarm_teleop_record.provenance import git_info
        data_dir = sup.data_dir()
        problem = md.data_dir_problem(data_dir) if opts.record else None
        if problem:
            raise_refused(sup, problem)
        if opts.record and prompt:
            ask_labels(sup, data_dir)
        labels, why = md.read_labels(data_dir) if opts.record else (None, None)
        disk = (record_profile or {}).get('disk') or {}
        items += [pf.disk_item(free_gb(data_dir), float(disk.get('min_free_gb_to_start', 0)),
                               data_dir, opts.record),
                  pf.labels_item(labels is not None, why, opts.record),
                  pf.commits_item(git_info(sup.checkout), git_info(robot_workspace()))]
    relay_here = opts.role == 'relay' or not opts.relay_url
    items.append(pf.adb_item(adb_facts(run_reverse=relay_here, port=opts.port) if relay_here
                             else None, relay_here))
    adb = items[-1]
    if relay_here and 'ran "' in adb.detail:
        sup.say(f'adb: {adb.detail}')
    return items


def raise_refused(sup, problem: str) -> None:
    """Stop before anything starts."""
    raise sv.Refused(f'refused: {problem}')


def ask_labels(sup: 'sv.Supervisor', data_dir: str) -> None:
    """Ask for the task label and operator; write them into the data directory only."""
    from ranger_xarm_teleop_record import metadata as md
    labels, _ = md.read_labels(data_dir)
    if labels is not None:
        answer = input('Task label and operator are set in the data directory. Enter keeps '
                       'them; type "new" to enter others: ')
        if answer.strip().lower() != 'new':
            return
    task = input('Task label: ').strip()
    operator = input('Operator: ').strip()
    if task and operator:
        md.write_labels(data_dir, task, operator)
        sup.say('labels written to the data directory')
    else:
        sup.say('labels left as they were (both are needed)')


# ------------------------------------------------------------------ stop and status
def stop_command(state_file: str, out=print, wait_s: float = STOP_SUPERVISOR_WAIT_S) -> int:
    """Stop the running session, also when its supervisor is gone."""
    state = procs.read_state(state_file)
    if state is None:
        out(f'no session is running here (no state file {state_file})')
        return sv.EXIT_OK
    if state.get('unreadable') or not state.get('mark'):
        out(f'the state file {state_file} is unreadable; cleared. No process could be found '
            'from it')
        procs.clear_state(state_file)
        return sv.EXIT_FAILED
    mark, name = state['mark'], state.get('session')
    live = procs.marked(mark)
    if not live:
        out(f'stale state file: session {name}, no process carries its mark. Cleared')
        procs.clear_state(state_file, mark)
        return sv.EXIT_OK
    sup_pid = state.get('supervisor_pid')
    if any(p.pid == sup_pid for p in live):
        out(f'asking the supervisor (pid {sup_pid}) of session {name} to stop')
        sv._signal_marked([sup_pid], signal.SIGINT, mark)
        end = time.monotonic() + wait_s
        while procs.mark_of(sup_pid) == mark and procs.alive(sup_pid):
            if time.monotonic() > end:
                out(f'the supervisor did not end within {wait_s:.0f} s')
                break
            time.sleep(0.2)
        report = os.path.join(state.get('log_dir') or '', 'stop_report.txt')
        if os.path.isfile(report):
            with open(report, encoding='utf-8') as handle:
                out(handle.read().rstrip())
        left = procs.marked(mark)
        if not left:
            procs.clear_state(state_file, mark)
            return sv.EXIT_OK
        out('processes of the session are still there; stopping them from the state file')
        notes: List[str] = []
        problems = sv.stop_processes(mark, state.get('children') or [], out, notes=notes)
        problems += sv.after_stop(state.get('session_dir'))
        out(sv.stop_report(state.get('session_dir'), problems, supervisor_gone=False,
                           notes=notes))
        procs.clear_state(state_file, mark)
        return sv.EXIT_FAILED
    out(f'the supervisor (pid {sup_pid}) of session {name} was already gone; stopping its '
        'processes from the state file and the session mark, in order')
    notes: List[str] = []
    problems = sv.stop_processes(mark, state.get('children') or [], out, notes=notes)
    problems += sv.after_stop(state.get('session_dir'))
    out(sv.stop_report(state.get('session_dir'), problems, supervisor_gone=True, notes=notes))
    procs.clear_state(state_file, mark)
    return sv.EXIT_FAILED if problems else sv.EXIT_OK


def status_command(state_file: str, out=print) -> int:
    """Say whether a session runs, and its phase."""
    state = procs.read_state(state_file)
    if state is None:
        out('no session')
        return sv.EXIT_OK
    if state.get('unreadable'):
        out(f'unreadable state file {state_file}')
        return sv.EXIT_FAILED
    live = procs.marked(state.get('mark') or '')
    sup = state.get('supervisor_pid')
    out(f'session {state.get("session")}: profile {state.get("profile")}, role '
        f'{state.get("role")}, phase {state.get("phase")}, {len(live)} processes, supervisor '
        f'{sup} {"running" if any(p.pid == sup for p in live) else "gone"}; logs '
        f'{state.get("log_dir")}')
    return sv.EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
