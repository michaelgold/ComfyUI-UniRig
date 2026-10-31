"""CPU executable-boundary tests, not model/GPU or rig-quality validation."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output'
    upstream = tmp_path / 'upstream'
    for directory in (input_dir, output_dir, upstream):
        directory.mkdir()
    (input_dir / 'mesh.glb').write_bytes(b'input fixture')
    monkeypatch.setitem(sys.modules, 'folder_paths', types.SimpleNamespace(
        get_input_directory=lambda: str(input_dir),
        get_output_directory=lambda: str(output_dir),
        get_temp_directory=lambda: str(tmp_path / 'temp'),
    ))
    monkeypatch.setenv('SKINTOKENS_ROOT', str(upstream))
    monkeypatch.setenv('SKINTOKENS_PYTHON', sys.executable)
    # Deliberate test doubles live only in the fixture's external checkout.
    (upstream / 'bottle.py').write_text('def run(*a, **kw): pass\n')
    (upstream / 'bpy_server.py').write_text('import time\ntime.sleep(60)\n')
    (upstream / 'demo.py').write_text('''
import json, os, struct
from pathlib import Path
MODEL_CKPTS = ['fixture.ckpt']
def wait_for_bpy_server(timeout=30): pass
def run_cli(args):
    Path('received.json').write_text(json.dumps(vars(args)))
    body = b'{"asset":{"version":"2.0"}}  '
    data = struct.pack('<4sII', b'glTF', 2, 20 + len(body))
    data += struct.pack('<I4s', len(body), b'JSON') + body
    Path(args.output).write_bytes(data)
''')
    spec = importlib.util.spec_from_file_location('skintokens_node', ROOT / 'nodes/skintokens.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, input_dir, output_dir, upstream


def test_real_worker_returns_download_and_forwards_options(runtime):
    module, _, output, upstream = runtime
    result = module.SkinTokensRigGLB().rig('mesh.glb', 'nested/rig.glb',
        use_skeleton=True, use_transfer=False, use_postprocess=True, num_beams=3)
    path = output / 'nested/rig.glb'
    assert result == {'result': (str(path),), 'ui': {'files': [
        {'filename': 'rig.glb', 'subfolder': 'nested', 'type': 'output'}]}}
    assert path.read_bytes().startswith(b'glTF')
    args = json.loads((upstream / 'received.json').read_text())
    assert args['use_skeleton'] is True
    assert args['use_transfer'] is False
    assert args['use_postprocess'] is True
    assert args['num_beams'] == 3
    assert Path(args['input']).is_absolute()
    assert args['output'] != str(path)  # fresh staging, never stale destination
    assert 'bpy' not in sys.modules


def test_schema_and_registration(runtime):
    import ast
    module, _, _, _ = runtime
    node = module.SkinTokensRigGLB
    schema = node.INPUT_TYPES()
    assert schema['required']['output_name'][1]['default'] == 'skintokens_rigged.glb'
    assert {key: value[1]['default'] for key, value in schema['optional'].items()} == {
        'use_skeleton': False, 'use_transfer': True, 'use_postprocess': False, 'num_beams': 10}
    assert node.RETURN_TYPES == ('STRING',)
    assert node.OUTPUT_NODE is True
    tree = ast.parse((ROOT / 'nodes/__init__.py').read_text())
    mappings = {statement.targets[0].id: [key.value for key in statement.value.keys]
                for statement in tree.body if isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Dict)}
    for name in ('NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'):
        assert 'SkinTokensRigGLB' in mappings[name]
        assert 'MIAAutoRig' in mappings[name]
        assert 'UniRigAutoRig' in mappings[name]


@pytest.mark.parametrize('name', ['../escape.glb', '/escape.glb', 'x/../../escape.glb',
                                   'x\\\\escape.glb', 'C:escape.glb', 'wrong.fbx', ''])
def test_output_traversal_rejected(runtime, name):
    module, _, _, _ = runtime
    with pytest.raises(ValueError):
        module.SkinTokensRigGLB().rig('mesh.glb', name)


def test_input_and_output_symlink_escapes(runtime, tmp_path):
    module, inputs, outputs, _ = runtime
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'mesh.glb').write_bytes(b'not in Comfy roots')
    (inputs / 'escape.glb').symlink_to(outside / 'mesh.glb')
    (outputs / 'escape').symlink_to(outside, target_is_directory=True)
    for source, destination in [('escape.glb', 'ok.glb'), ('mesh.glb', 'escape/rig.glb'),
                                (str(outside / 'mesh.glb'), 'ok.glb')]:
        with pytest.raises(ValueError):
            module.SkinTokensRigGLB().rig(source, destination)
    with pytest.raises(ValueError):
        module.SkinTokensRigGLB().rig('../outside/mesh.glb', 'ok.glb')


def test_existing_output_untouched(runtime):
    module, _, output, upstream = runtime
    (output / 'old.glb').write_bytes(b'old')
    with pytest.raises(FileExistsError):
        module.SkinTokensRigGLB().rig('mesh.glb', 'old.glb')
    assert (output / 'old.glb').read_bytes() == b'old'
    assert not (upstream / 'received.json').exists()


@pytest.mark.parametrize('body, match', [
    ("raise RuntimeError('real child diagnostic')", 'real child diagnostic'),
    ('pass', 'fresh GLB'),
    ("Path(args.output).write_bytes(b'')", 'empty or truncated'),
    ("Path(args.output).write_bytes(b'X' * 32)", 'valid GLB'),
    ("Path(args.output).symlink_to(args.input)", 'fresh GLB'),
])
def test_subprocess_failure_or_bad_output_never_published(runtime, body, match):
    module, _, output, upstream = runtime
    (upstream / 'demo.py').write_text(
        "from pathlib import Path\nMODEL_CKPTS=['fixture']\n"
        "def wait_for_bpy_server(timeout=30): pass\n"
        "def run_cli(args):\n    " + body + '\n')
    with pytest.raises(RuntimeError, match=match):
        module.SkinTokensRigGLB().rig('mesh.glb', 'bad.glb')
    assert not (output / 'bad.glb').exists()
    assert list(output.iterdir()) == []


def test_timeout_kills_worker_and_descendants(runtime, monkeypatch):
    import os
    import time
    module, _, output, upstream = runtime
    (upstream / 'demo.py').write_text('''
import os, time
from pathlib import Path
MODEL_CKPTS=['fixture']
def wait_for_bpy_server(timeout=30):
    while not Path('server.pid').exists(): time.sleep(.01)
def run_cli(args):
    Path('worker.pid').write_text(str(os.getpid()))
    time.sleep(60)
''')
    (upstream / 'bpy_server.py').write_text(
        "import os, time\nfrom pathlib import Path\n"
        "Path('server.pid').write_text(str(os.getpid()))\ntime.sleep(60)\n")
    monkeypatch.setenv('SKINTOKENS_TIMEOUT_SECONDS', '0.5')
    with pytest.raises(RuntimeError, match='timed out'):
        module.SkinTokensRigGLB().rig('mesh.glb', 'timed.glb')
    for name in ('worker.pid', 'server.pid'):
        pid = int((upstream / name).read_text())
        for _ in range(100):
            stat = Path(f'/proc/{pid}/stat')
            if not stat.exists() or stat.read_text().split()[2] == 'Z':
                break
            time.sleep(.01)
        else:
            pytest.fail(f'Child {pid} still running after timeout')
    assert list(output.iterdir()) == []


def test_missing_configuration_and_bad_interpreter(runtime, monkeypatch):
    module, _, _, upstream = runtime
    monkeypatch.delenv('SKINTOKENS_ROOT')
    with pytest.raises(RuntimeError, match='SKINTOKENS_ROOT'):
        module.SkinTokensRigGLB().rig('mesh.glb')
    monkeypatch.setenv('SKINTOKENS_ROOT', str(upstream))
    monkeypatch.setenv('SKINTOKENS_PYTHON', '/nonexistent/skintokens-python')
    with pytest.raises(RuntimeError, match='executable Python'):
        module.SkinTokensRigGLB().rig('mesh.glb')


def test_default_options_and_environment_isolation(runtime, monkeypatch):
    module, inputs, outputs, upstream = runtime
    demo = upstream / 'demo.py'
    demo.write_text(demo.read_text() + '''
_original = run_cli
def run_cli(args):
    assert 'PYTHONPATH' not in os.environ
    assert 'PYTHONHOME' not in os.environ
    assert 'LD_PRELOAD' not in os.environ
    _original(args)
''')
    for key in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD'):
        monkeypatch.setenv(key, '/not/a/real/library')
    result = module.SkinTokensRigGLB().rig(str(inputs / 'mesh.glb'))
    assert result['ui']['files'][0]['subfolder'] == ''
    args = json.loads((upstream / 'received.json').read_text())
    assert args['use_transfer'] is True
    assert args['use_skeleton'] is False
    assert args['use_postprocess'] is False
    assert args['num_beams'] == 10
    assert (outputs / 'skintokens_rigged.glb').is_file()


@pytest.mark.parametrize('timeout', ['0', '-1', 'nan', 'inf', 'not-a-number'])
def test_invalid_timeout_rejected(runtime, monkeypatch, timeout):
    module, _, _, upstream = runtime
    monkeypatch.setenv('SKINTOKENS_TIMEOUT_SECONDS', timeout)
    with pytest.raises(ValueError):
        module.SkinTokensRigGLB().rig('mesh.glb')
    assert not (upstream / 'received.json').exists()


def test_concurrent_checkout_lock_fails_closed(runtime):
    import fcntl
    module, _, _, upstream = runtime
    with (upstream / '.comfy-skintokens.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='Another SkinTokens job'):
            module.SkinTokensRigGLB().rig('mesh.glb')


def test_occupied_upstream_port_fails_closed(runtime):
    import socket
    module, _, output, _ = runtime
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', 59876))
        listener.listen()
        with pytest.raises(RuntimeError, match='port 59876 is occupied'):
            module.SkinTokensRigGLB().rig('mesh.glb')
    assert list(output.iterdir()) == []


@pytest.mark.skipif(not Path('/proc/net/tcp').exists(), reason='Linux socket-table assertion')
def test_real_http_server_loopback_and_cleanup(runtime, monkeypatch):
    import socket
    module, _, _, upstream = runtime
    monkeypatch.setenv('http_proxy', 'http://untrusted-proxy.invalid:8080')
    monkeypatch.setenv('no_proxy', '')
    # Exercise real Bottle/WSGI and an upstream-shaped wildcard/threaded entry;
    # only model/Blender work is replaced, not the transport or worker lifecycle.
    (upstream / 'bottle.py').unlink()
    (upstream / 'bpy_server.py').write_text('''
import bottle, os, threading
from pathlib import Path
app = bottle.Bottle()
@app.get('/ping')
def ping(): return 'pong'
@app.route('/load', method='GET')
@app.route('/export', method='POST')
@app.route('/transfer', method='POST')
def payload():
    bottle.response.content_type = 'application/octet-stream'
    return bottle.request.body.read()
# The actual upstream invokes this from a thread with host=0.0.0.0.
threading.Thread(target=lambda: bottle.run(app, host='0.0.0.0', port=59876,
                                         server='tornado'), daemon=False).start()
Path('server.pid').write_text(str(os.getpid()))
''')
    demo = upstream / 'demo.py'
    demo.write_text(demo.read_text() + '''
import requests, time, socket
def wait_for_bpy_server(timeout=30):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            response = requests.get('http://127.0.0.1:59876/ping', timeout=.2)
            if response.status_code == 200 and response.text == 'pong': return
        except requests.RequestException: pass
        time.sleep(.01)
    raise RuntimeError('real HTTP server not ready')
_original = run_cli
def run_cli(args):
    # Linux listener table confirms the address rather than trusting kwargs.
    rows = Path('/proc/net/tcp').read_text().splitlines()[1:]
    listeners = [row.split()[1] for row in rows
                 if row.split()[3] == '0A' and row.split()[1].endswith(':E9E4')]
    assert listeners == ['0100007F:E9E4'], listeners
    for method, endpoint in (('GET', 'load'), ('POST', 'export'), ('POST', 'transfer')):
        payload = bytes(range(256)) * 1024
        response = requests.request(method, f'http://localhost:59876/{endpoint}',
                                    data=payload, timeout=2)
        assert response.status_code == 200 and response.content == payload
    _original(args)
''')
    module.SkinTokensRigGLB().rig('mesh.glb', 'one.glb')
    # A second job must really start, not share the previous server or encounter
    # leaked ownership/TIME_WAIT state from the first job.
    module.SkinTokensRigGLB().rig('mesh.glb', 'two.glb')
    with socket.socket() as connection:
        assert connection.connect_ex(('127.0.0.1', 59876)) != 0


def test_port_remains_reserved_during_upstream_import(runtime):
    module, _, _, upstream = runtime
    demo = upstream / 'demo.py'
    demo.write_text('''
import socket
with socket.socket() as contender:
    contender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        contender.bind(('127.0.0.1', 59876))
        contender.listen()
    except OSError:
        pass
    else:
        raise AssertionError('port reservation lost before upstream import')
''' + demo.read_text())
    module.SkinTokensRigGLB().rig('mesh.glb')


def test_pickle_transport_bypasses_inherited_proxies(runtime, monkeypatch):
    module, _, _, upstream = runtime
    for key in ('HTTP_PROXY', 'http_proxy', 'ALL_PROXY', 'all_proxy'):
        monkeypatch.setenv(key, 'http://untrusted-proxy.invalid:8080')
    for key in ('NO_PROXY', 'no_proxy'):
        monkeypatch.setenv(key, 'unrelated.example')
    demo = upstream / 'demo.py'
    demo.write_text(demo.read_text() + '''
_original = run_cli
def run_cli(args):
    from requests.utils import get_environ_proxies
    for host in ('localhost', '127.0.0.1', '[::1]'):
        assert not get_environ_proxies(f'http://{host}:59876/load'), 'pickle proxy leak'
    _original(args)
''')
    module.SkinTokensRigGLB().rig('mesh.glb')
