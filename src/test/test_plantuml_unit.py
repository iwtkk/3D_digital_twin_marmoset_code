import sys
import unittest
from pathlib import Path

from src import plantuml


class Dummy:
    def public_method(self) -> str:
        return "ok"

    def _private_method(self) -> str:
        return "no"


class TestPlantuml(unittest.TestCase):
    def test_build_class_uml_includes_public_methods(self):
        uml = plantuml.build_class_uml(sys.modules[__name__])
        self.assertIn("class Dummy", uml)
        self.assertIn("+public_method()", uml)
        self.assertNotIn("_private_method", uml)

    def test_build_sequence_uml_renders_events(self):
        events = [
            {
                "src": "ROOT",
                "dst": "build_class_uml",
                "action": "enter",
                "detail": "start",
            }
        ]
        uml = plantuml.build_sequence_uml(events)
        self.assertIn("actor ROOT", uml)
        self.assertIn('participant "build_class_uml"', uml)
        self.assertIn("ROOT -> build_class_uml", uml)

    def test_build_activity_uml_renders_enter_events(self):
        events = [{"src": "ROOT", "dst": "build_sequence_uml", "action": "enter"}]
        uml = plantuml.build_activity_uml(events)
        self.assertIn(":build_sequence_uml;", uml)

    def test_write_uml_files_outputs_pu(self):
        project_root = Path(__file__).resolve().parents[1]
        output_root = project_root / "agent" / "uml"
        file_stem = Path(plantuml.__file__).stem
        events = [
            {
                "src": "ROOT",
                "dst": "build_class_uml",
                "action": "enter",
                "detail": "start",
            }
        ]
        paths = plantuml.write_uml_files(plantuml, output_root, file_stem, events=events)
        self.assertTrue(paths["class"].exists())
        self.assertTrue(paths["sequence"].exists())
        self.assertTrue(paths["activity"].exists())
        self.assertIn("@startuml", paths["sequence"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
