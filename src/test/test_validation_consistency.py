"""
scripts/06_validation/consistency/ および src/marmo3d/validation/utils.py の
ユニットテスト。

テスト対象:
  - marmo3d.validation.utils の各ヘルパー関数
  - config/consistency.yaml のパステンプレートが consistency/ を含むこと
  - run_pipeline.py の build_pipeline が正しい出力パスを生成すること
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

# プロジェクトルートを解決して src/ を sys.path に追加
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from marmo3d.validation.utils import (
    build_points2d_for_frame,
    cam_map,
    expand_path_template,
    load_ids_from_yaml,
    rodrigues_to_R,
)


# ---------------------------------------------------------------------------
# rodrigues_to_R
# ---------------------------------------------------------------------------

class TestRodriguesToR(unittest.TestCase):
    def test_zero_vector_returns_identity(self):
        R = rodrigues_to_R(np.zeros(3))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90deg_rotation_around_z(self):
        # Z 軸まわり 90 度 = [0, 0, π/2]
        rvec = np.array([0.0, 0.0, math.pi / 2])
        R = rodrigues_to_R(rvec)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_output_is_proper_rotation_matrix(self):
        rvec = np.array([0.3, -0.5, 0.8])
        R = rodrigues_to_R(rvec)
        # R^T R = I かつ det(R) = 1
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=12)

    def test_input_shape_1d_3(self):
        rvec = np.array([0.1, 0.2, 0.3])
        R = rodrigues_to_R(rvec)
        self.assertEqual(R.shape, (3, 3))

    def test_input_accepts_reshaped(self):
        rvec = np.array([[0.1], [0.2], [0.3]])
        R = rodrigues_to_R(rvec)
        self.assertEqual(R.shape, (3, 3))


# ---------------------------------------------------------------------------
# build_points2d_for_frame
# ---------------------------------------------------------------------------

class TestBuildPoints2dForFrame(unittest.TestCase):
    def _make_kp2d_f(self, n_frame=10, n_kp=5, n_animal=2, n_cam=3):
        rng = np.random.default_rng(0)
        arr = rng.random((n_frame, n_kp, n_animal, 3, n_cam))
        arr[:, :, :, 2, :] = rng.random((n_frame, n_kp, n_animal, n_cam))
        return arr

    def test_output_shapes(self):
        kp2d_f = self._make_kp2d_f(n_frame=10, n_kp=5, n_animal=2, n_cam=3)
        pts, scores = build_points2d_for_frame(kp2d_f, frame=0, animal=0, score_thresh=0.3)
        self.assertEqual(pts.shape, (3, 5, 2))
        self.assertEqual(scores.shape, (3, 5))

    def test_low_score_becomes_nan(self):
        kp2d_f = np.ones((1, 1, 1, 3, 1))
        kp2d_f[0, 0, 0, 2, 0] = 0.1  # score < thresh
        pts, scores = build_points2d_for_frame(kp2d_f, frame=0, animal=0, score_thresh=0.5)
        self.assertTrue(np.isnan(pts[0, 0, 0]))

    def test_high_score_not_nan(self):
        kp2d_f = np.ones((1, 1, 1, 3, 1))
        kp2d_f[0, 0, 0, 2, 0] = 0.9  # score > thresh
        pts, _ = build_points2d_for_frame(kp2d_f, frame=0, animal=0, score_thresh=0.5)
        self.assertFalse(np.isnan(pts[0, 0, 0]))

    def test_frame_out_of_range_raises(self):
        kp2d_f = self._make_kp2d_f(n_frame=5)
        with self.assertRaises(IndexError):
            build_points2d_for_frame(kp2d_f, frame=5, animal=0, score_thresh=0.3)

    def test_animal_out_of_range_raises(self):
        kp2d_f = self._make_kp2d_f(n_animal=2)
        with self.assertRaises(IndexError):
            build_points2d_for_frame(kp2d_f, frame=0, animal=2, score_thresh=0.3)


# ---------------------------------------------------------------------------
# cam_map
# ---------------------------------------------------------------------------

class TestCamMap(unittest.TestCase):
    def test_has_eight_cameras(self):
        self.assertEqual(len(cam_map), 8)

    def test_labels_are_cam_1_to_8(self):
        labels = set(cam_map.values())
        self.assertEqual(labels, {f"Cam {i}" for i in range(1, 9)})

    def test_keys_are_integers(self):
        for k in cam_map:
            self.assertIsInstance(k, int)


# ---------------------------------------------------------------------------
# load_ids_from_yaml
# ---------------------------------------------------------------------------

class TestLoadIdsFromYaml(unittest.TestCase):
    def test_returns_string_list(self):
        data = {"camera_id": [23506226, 23511607, 23506214]}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp = f.name
        ids = load_ids_from_yaml(tmp)
        self.assertEqual(ids, ["23506226", "23511607", "23506214"])

    def test_already_string_ids(self):
        data = {"camera_id": ["a", "b"]}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            tmp = f.name
        ids = load_ids_from_yaml(tmp)
        self.assertEqual(ids, ["a", "b"])


# ---------------------------------------------------------------------------
# consistency.yaml パステンプレートに consistency/ が含まれること
# ---------------------------------------------------------------------------

class TestConsistencyYamlPaths(unittest.TestCase):
    def setUp(self):
        config_path = _PROJECT_ROOT / "config" / "consistency.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self._cfg = yaml.safe_load(f)

    def test_frame_extraction_output_root_contains_consistency(self):
        root = self._cfg["validation"]["frame_extraction"]["output_root"]
        self.assertIn("consistency", root, f"frame_extraction.output_root should contain 'consistency': {root}")

    def test_render_root_contains_consistency(self):
        root = self._cfg["validation"]["render"]["root"]
        self.assertIn("consistency", root, f"render.root should contain 'consistency': {root}")

    def test_metrics_output_path_contains_consistency(self):
        path = self._cfg["validation"]["metrics"]["output_path"]
        self.assertIn("consistency", path, f"metrics.output_path should contain 'consistency': {path}")

    def test_expand_path_template_render(self):
        template = self._cfg["validation"]["render"]["root"]
        resolved = expand_path_template(template, _PROJECT_ROOT)
        self.assertIn("consistency", str(resolved))
        self.assertTrue(str(resolved).endswith("render") or "render" in str(resolved))


# ---------------------------------------------------------------------------
# run_pipeline.build_pipeline の出力パスに consistency/ が含まれること
# ---------------------------------------------------------------------------

class TestRunPipelinePaths(unittest.TestCase):
    def _import_build_pipeline(self):
        import importlib.util
        script_path = (
            _PROJECT_ROOT / "scripts" / "06_validation" / "consistency" / "run_pipeline.py"
        )
        spec = importlib.util.spec_from_file_location("run_pipeline", script_path)
        mod = importlib.util.module_from_spec(spec)
        # sys.modules に登録してから exec しないと dataclass の型解決が失敗する
        sys.modules["run_pipeline"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_output_dir_contains_consistency(self):
        mod = self._import_build_pipeline()
        config = mod.PipelineConfig()
        steps = mod.build_pipeline(config)
        # Step 3（extract_frames）の --output-root 引数を確認
        extract_step = next(s for s in steps if "Extract" in s.name)
        cmd = " ".join(extract_step.command)
        self.assertIn("consistency", cmd, f"extract step should output to consistency/: {cmd}")

    def test_render_output_contains_consistency(self):
        mod = self._import_build_pipeline()
        config = mod.PipelineConfig()
        steps = mod.build_pipeline(config)
        # Step 2（render）の --output-root 引数を確認
        render_step = next(s for s in steps if "Render" in s.name)
        cmd = " ".join(render_step.command)
        self.assertIn("consistency", cmd, f"render step should output to consistency/: {cmd}")


if __name__ == "__main__":
    unittest.main()
