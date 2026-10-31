"""Private POSIX worker for SkinTokens demo.py at 273b691d.

Keep this stdlib-only until main(): all upstream imports belong to the child.
The upstream server has a fixed port and uses trusted-local pickle payloads.
"""
import argparse
import fcntl
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys


def main():
    # Serialized torch/pickle traffic must never traverse an inherited proxy.
    # requests prefers lowercase no_proxy; set both, before upstream imports and
    # before spawning Blender/dataloader children, while preserving other bypasses.
    bypass = ','.join((os.environ.get('NO_PROXY', ''),
                       os.environ.get('no_proxy', ''), 'localhost,127.0.0.1,::1'))
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = bypass
    # The node sets cwd to SKINTOKENS_ROOT, not the custom-node directory.
    sys.path.insert(0, str(Path.cwd()))
    if len(sys.argv) == 3 and sys.argv[1] == '--server':
        import bottle
        original_run = bottle.run
        listener = socket.socket(fileno=int(sys.argv[2]))
        listener.set_inheritable(False)

        def local_run(*args, **kwargs):
            # Adopt the already bound listener: never release/rebind the fixed
            # port between the ownership check and serving pickle requests.
            class LocalServer(bottle.ServerAdapter):
                def run(self, handler):
                    from wsgiref.simple_server import WSGIServer, WSGIRequestHandler
                    httpd = WSGIServer(listener.getsockname(), WSGIRequestHandler,
                                      bind_and_activate=False)
                    httpd.socket.close()
                    httpd.socket = listener
                    httpd.server_name = 'localhost'
                    httpd.server_port = listener.getsockname()[1]
                    httpd.setup_environ()
                    httpd.set_app(handler)
                    try:
                        httpd.serve_forever()
                    finally:
                        httpd.server_close()

            kwargs['host'] = '127.0.0.1'
            kwargs['server'] = LocalServer
            kwargs['reloader'] = False
            return original_run(*args, **kwargs)

        bottle.run = local_run
        runpy.run_path('bpy_server.py', run_name='__main__')
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_beams', type=int, default=10)
    for flag in ('use_skeleton', 'use_transfer', 'use_postprocess'):
        parser.add_argument('--' + flag, action='store_true')
    args = parser.parse_args()
    # Fail concurrent runs rather than sharing upstream's mutable bpy scene.
    with Path('.comfy-skintokens.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another SkinTokens job is using this checkout') from exc
        with socket.socket() as listener:
            # Allow sequential jobs after HTTP TIME_WAIT, never SO_REUSEPORT:
            # bind+listen still excludes another live service across checkouts.
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(('127.0.0.1', 59876))
                listener.listen(16)
            except OSError as exc:
                raise RuntimeError('SkinTokens bpy port 59876 is occupied; stop the other server') from exc
            import demo
            args.top_k = 5
            args.top_p = 0.95
            args.temperature = 1.0
            args.repetition_penalty = 2.0
            args.model_ckpt = demo.MODEL_CKPTS[0]
            args.hf_path = None
            # Inherit the process group and listener, not a new session. The
            # parent closes its listener copy before dataloader workers spawn.
            server = subprocess.Popen(
                [sys.executable, '-u', str(Path(__file__).resolve()),
                 '--server', str(listener.fileno())], pass_fds=(listener.fileno(),))
        try:
            demo.wait_for_bpy_server(timeout=30)
            if server.poll() is not None:
                raise RuntimeError(f'SkinTokens bpy server exited: {server.returncode}')
            demo.run_cli(args)
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == '__main__':
    main()
