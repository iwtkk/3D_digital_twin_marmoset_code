import os
import pickle
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from marmo3d.blender import render_setting
from src import plantuml
from src import tracer as tracer_module

TRACE_DEFAULT = os.environ.get("TESTBLENDER_TRACE", "1") not in ("0", "false", "False")
TRACE_PRINT = os.environ.get("TESTBLENDER_TRACE_PRINT", "1") not in ("0", "false", "False")
render_setting.set_trace_enabled(TRACE_DEFAULT, print_steps=TRACE_PRINT)


class TestSplitFrames(unittest.TestCase):
    def test_split_frames_basic(self):
        frames = [1, 2, 3, 4, 5]
        self.assertEqual(render_setting.split_frames(frames, 2), [[1, 2, 3], [4, 5]])

    def test_split_frames_empty(self):
        self.assertEqual(render_setting.split_frames([], 3), [])

    def test_split_frames_invalid_split(self):
        with self.assertRaises(ValueError):
            render_setting.split_frames([1], 0)


class TestResolveFrames(unittest.TestCase):
    def _write_pickle(self, path: Path, payload) -> None:
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def test_resolve_frames_from_frame_spec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Path(tmpdir)
            cases = [
                (5, [5]),
                ((1, 3), [1, 2, 3]),
                ([3, 1, 2, 2], [1, 2, 3]),
            ]
            for frames, expected in cases:
                with self.subTest(frames=frames):
                    self.assertEqual(render_setting.resolve_frames(session, frames), expected)

    def test_resolve_frames_none_uses_kp3d_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Path(tmpdir)
            self._write_pickle(session / "kp3d.pickle", [0, 0, 0])
            self.assertEqual(render_setting.resolve_frames(session, None), [0, 1, 2])

    def test_resolve_frames_empty_list_uses_kp3d_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Path(tmpdir)
            self._write_pickle(session / "kp3d.pickle", [0, 0])
            self.assertEqual(render_setting.resolve_frames(session, []), [0, 1])

    def test_resolve_frames_uses_kp3d_fxd_joint_len_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            session = Path(tmpdir)
            self._write_pickle(session / "kp3d_fxdJointLen.pickle", [0, 0, 0, 0])
            self.assertEqual(render_setting.resolve_frames(session, None), [0, 1, 2, 3])


class TestOverridesSkip(unittest.TestCase):
    def test_should_skip_output_overrides_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            self.assertFalse(render_setting.should_skip_output(True, output_dir))

    def test_should_skip_output_overrides_false_with_existing_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            self.assertTrue(render_setting.should_skip_output(False, output_dir))

    def test_should_skip_output_overrides_false_without_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            self.assertFalse(render_setting.should_skip_output(False, output_dir))


class DummyPool:
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.submitted = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, iterable):
        items = list(iterable)
        self.submitted.extend(items)
        return [func(item) for item in items]

    def submit(self, func, *args, **kwargs):
        self.submitted.append(args[0] if args else None)
        result = mock.Mock()
        result.result.return_value = func(*args, **kwargs)
        return result


class TestExecutor(unittest.TestCase):
    def test_executor_retries_on_timeout_then_success(self):
        executor = render_setting.Executor()
        executor.workers = 1
        executor.retries = 2
        executor.timeout = 0.1
        commands = [["echo", "ok"]]

        timeout_exc = subprocess.TimeoutExpired(cmd=commands[0], timeout=executor.timeout)
        success = subprocess.CompletedProcess(args=commands[0], returncode=0)

        with mock.patch("subprocess.run", side_effect=[timeout_exc, success]) as run:
            executor.execute(commands, dry_run=False)
            self.assertEqual(run.call_count, 2)

    def test_executor_exhausts_retries_on_timeout(self):
        executor = render_setting.Executor()
        executor.workers = 1
        executor.retries = 1
        executor.timeout = 0.1
        commands = [["echo", "ng"]]

        timeout_exc = subprocess.TimeoutExpired(cmd=commands[0], timeout=executor.timeout)
        with mock.patch("subprocess.run", side_effect=timeout_exc) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                executor.execute(commands, dry_run=False)
            self.assertEqual(run.call_count, 2)

    def test_executor_single_worker_runs_in_order(self):
        executor = render_setting.Executor()
        executor.workers = 1
        executor.retries = 0
        executor.timeout = None
        commands = [["echo", "1"], ["echo", "2"]]

        success = subprocess.CompletedProcess(args=commands[0], returncode=0)
        with mock.patch("subprocess.run", return_value=success) as run:
            executor.execute(commands, dry_run=False)
            self.assertEqual(run.call_count, 2)
            called = [call.args[0] for call in run.call_args_list]
            self.assertEqual(called, commands)

    def test_executor_parallel_uses_thread_pool(self):
        executor = render_setting.Executor()
        executor.workers = 2
        executor.retries = 0
        executor.timeout = None
        commands = [["echo", "1"], ["echo", "2"]]

        success = subprocess.CompletedProcess(args=commands[0], returncode=0)
        with mock.patch(
            "concurrent.futures.ThreadPoolExecutor",
            side_effect=lambda max_workers: DummyPool(max_workers),
        ) as pool_cls:
            with mock.patch("subprocess.run", return_value=success) as run:
                executor.execute(commands, dry_run=False)
                self.assertEqual(run.call_count, 2)

        call_args = pool_cls.call_args
        if call_args.kwargs:
            max_workers = call_args.kwargs.get("max_workers")
        else:
            max_workers = call_args.args[0]
        self.assertEqual(max_workers, 2)


def tearDownModule():
    if os.environ.get("TESTBLENDER_TRACE_DUMP", "0") not in ("0", "false", "False"):
        project_root = Path(__file__).resolve().parents[1]
        output_root = project_root / "agent" / "uml"
        file_stem = Path(render_setting.__file__).stem
        plantuml.write_uml_files(
            render_setting,
            output_root,
            file_stem,
            events=tracer_module.trace_events(),
        )


if __name__ == "__main__":
    unittest.main()
