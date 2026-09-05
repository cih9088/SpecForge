"""Regression tests for ordered SGLang patch transitions."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "apply_sglang_spec_capture_patch.sh"


def _base_patch(base: str, patched: str, sink: str) -> str:
    return f"""\
diff --git a/python/sglang/srt/example.py b/python/sglang/srt/example.py
--- a/python/sglang/srt/example.py
+++ b/python/sglang/srt/example.py
@@ -1 +1 @@
-{base}
+{patched}
diff --git a/python/sglang/srt/spec_capture_sink.py b/python/sglang/srt/spec_capture_sink.py
new file mode 100644
--- /dev/null
+++ b/python/sglang/srt/spec_capture_sink.py
@@ -0,0 +1 @@
+{sink}
"""


def _addon_patch(base: str, patched: str) -> str:
    return f"""\
diff --git a/python/sglang/srt/addon.py b/python/sglang/srt/addon.py
--- a/python/sglang/srt/addon.py
+++ b/python/sglang/srt/addon.py
@@ -1 +1 @@
-{base}
+{patched}
"""


class ApplySglangSpecCapturePatchTest(unittest.TestCase):
    def setUp(self) -> None:
        # Keep the fake site-packages below this Git worktree. The helper must
        # prevent git apply from discovering the parent repository and
        # silently skipping virtualenv paths.
        self.temp_dir = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp_dir.cleanup)
        self.site_packages = Path(self.temp_dir.name) / "site-packages"
        self.srt = self.site_packages / "sglang" / "srt"
        self.srt.mkdir(parents=True)

        self.example = self.srt / "example.py"
        self.addon = self.srt / "addon.py"
        self.sink = self.srt / "spec_capture_sink.py"
        self.example.write_text("new base\n", encoding="utf-8")
        self.addon.write_text("addon base\n", encoding="utf-8")

        self.base_record = self.site_packages / "sglang" / ".spec_capture_patch.applied"
        self.addon_record = (
            self.site_packages
            / "sglang"
            / ".spec_capture_patch.qwen3.5-eagle3.patch.applied"
        )
        self.order = self.site_packages / "sglang" / ".spec_capture_patch.order"

        self.base_patch = Path(self.temp_dir.name) / "spec-capture.patch"
        self.base_patch.write_text(
            _base_patch("new base", "new patched", "new sink"),
            encoding="utf-8",
        )
        self.addon_patch = Path(self.temp_dir.name) / "qwen3.5-eagle3.patch"
        self.addon_patch.write_text(
            _addon_patch("addon base", "addon patched"), encoding="utf-8"
        )

        self.env = {
            **os.environ,
            "SPECFORGE_SGLANG_ROOT": str(self.site_packages),
            "SPECFORGE_SGLANG_VERSION": "0.5.18",
        }

    def run_script(
        self, *args: str, patch: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = {
            **self.env,
            "SPECFORGE_SPEC_CAPTURE_PATCH": str(patch or self.base_patch),
        }
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            check=False,
            capture_output=True,
            env=env,
            text=True,
        )

    def apply_stack(self) -> None:
        base_result = self.run_script()
        self.assertEqual(base_result.returncode, 0, base_result.stderr)
        addon_result = self.run_script(patch=self.addon_patch)
        self.assertEqual(addon_result.returncode, 0, addon_result.stderr)

    def assert_order(self, *names: str) -> None:
        if names:
            self.assertEqual(
                self.order.read_text(encoding="utf-8"),
                "".join(f"{name}\n" for name in names),
            )
        else:
            self.assertFalse(self.order.exists())

    def seed_cached_upgrade(self, installed_source: str = "new base") -> str:
        old_patch = _base_patch("old base", "old patched", "old sink")
        self.example.write_text(f"{installed_source}\n", encoding="utf-8")
        self.sink.write_text("old sink\n", encoding="utf-8")
        self.base_record.write_text(old_patch, encoding="utf-8")
        return old_patch

    def test_recovers_files_left_by_pip_upgrade_and_is_idempotent(self) -> None:
        self.seed_cached_upgrade()

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recovered stale spec-capture.patch files", result.stdout)
        self.assertEqual(self.example.read_text(encoding="utf-8"), "new patched\n")
        self.assertEqual(self.sink.read_text(encoding="utf-8"), "new sink\n")
        self.assertEqual(
            self.base_record.read_text(encoding="utf-8"),
            self.base_patch.read_text(encoding="utf-8"),
        )
        self.assert_order("spec-capture.patch")

        second_result = self.run_script()
        self.assertEqual(second_result.returncode, 0, second_result.stderr)
        self.assertIn("already applied", second_result.stdout)

    def test_restores_stale_sink_when_new_patch_does_not_apply(self) -> None:
        old_patch = self.seed_cached_upgrade(installed_source="unexpected source")

        result = self.run_script()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown spec-capture patch state", result.stderr)
        self.assertEqual(
            self.example.read_text(encoding="utf-8"), "unexpected source\n"
        )
        self.assertEqual(self.sink.read_text(encoding="utf-8"), "old sink\n")
        self.assertEqual(self.base_record.read_text(encoding="utf-8"), old_patch)

    def test_applies_two_patches_in_order_and_reruns_idempotently(self) -> None:
        self.apply_stack()

        self.assertEqual(self.example.read_text(encoding="utf-8"), "new patched\n")
        self.assertEqual(self.addon.read_text(encoding="utf-8"), "addon patched\n")
        self.assertEqual(self.sink.read_text(encoding="utf-8"), "new sink\n")
        self.assertTrue(self.base_record.exists())
        self.assertTrue(self.addon_record.exists())
        self.assert_order("spec-capture.patch", "qwen3.5-eagle3.patch")

        base_result = self.run_script()
        addon_result = self.run_script(patch=self.addon_patch)
        self.assertEqual(base_result.returncode, 0, base_result.stderr)
        self.assertEqual(addon_result.returncode, 0, addon_result.stderr)
        self.assertIn("already applied", base_result.stdout)
        self.assertIn("already applied", addon_result.stdout)
        self.assert_order("spec-capture.patch", "qwen3.5-eagle3.patch")

    def test_rejects_addon_before_base_without_mutating_tree(self) -> None:
        result = self.run_script(patch=self.addon_patch)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("apply spec-capture.patch first", result.stderr)
        self.assertEqual(self.addon.read_text(encoding="utf-8"), "addon base\n")
        self.assertFalse(self.addon_record.exists())
        self.assert_order()

    def test_enforces_reverse_order_and_uses_recorded_patch(self) -> None:
        self.apply_stack()

        base_first = self.run_script("--reverse")
        self.assertNotEqual(base_first.returncode, 0)
        self.assertIn("reverse qwen3.5-eagle3.patch first", base_first.stderr)
        self.assertEqual(self.example.read_text(encoding="utf-8"), "new patched\n")
        self.assertEqual(self.addon.read_text(encoding="utf-8"), "addon patched\n")

        self.addon_patch.write_text(
            _addon_patch("different base", "different patched"), encoding="utf-8"
        )
        addon_result = self.run_script("--reverse", patch=self.addon_patch)
        self.assertEqual(addon_result.returncode, 0, addon_result.stderr)
        self.assertEqual(self.addon.read_text(encoding="utf-8"), "addon base\n")
        self.assertFalse(self.addon_record.exists())
        self.assert_order("spec-capture.patch")

        base_result = self.run_script("--reverse")
        self.assertEqual(base_result.returncode, 0, base_result.stderr)
        self.assertEqual(self.example.read_text(encoding="utf-8"), "new base\n")
        self.assertFalse(self.sink.exists())
        self.assertFalse(self.base_record.exists())
        self.assert_order()

    def test_updates_only_the_top_patch(self) -> None:
        self.apply_stack()
        self.addon_patch.write_text(
            _addon_patch("addon base", "addon v2"), encoding="utf-8"
        )

        addon_result = self.run_script(patch=self.addon_patch)

        self.assertEqual(addon_result.returncode, 0, addon_result.stderr)
        self.assertIn("updated", addon_result.stdout)
        self.assertEqual(self.addon.read_text(encoding="utf-8"), "addon v2\n")
        self.assertEqual(self.example.read_text(encoding="utf-8"), "new patched\n")
        self.assert_order("spec-capture.patch", "qwen3.5-eagle3.patch")

        self.base_patch.write_text(
            _base_patch("new base", "base v2", "new sink"), encoding="utf-8"
        )
        base_result = self.run_script()
        self.assertNotEqual(base_result.returncode, 0)
        self.assertIn("reverse patches above", base_result.stderr)
        self.assertEqual(self.example.read_text(encoding="utf-8"), "new patched\n")

    def test_rolls_back_failed_top_patch_update(self) -> None:
        self.apply_stack()
        recorded = self.addon_record.read_text(encoding="utf-8")
        self.addon_patch.write_text(
            _addon_patch("unexpected base", "addon v2"), encoding="utf-8"
        )

        result = self.run_script(patch=self.addon_patch)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("restored the recorded version", result.stderr)
        self.assertEqual(self.addon.read_text(encoding="utf-8"), "addon patched\n")
        self.assertEqual(self.addon_record.read_text(encoding="utf-8"), recorded)
        self.assert_order("spec-capture.patch", "qwen3.5-eagle3.patch")

    def test_cached_upgrade_recovery_must_run_in_application_order(self) -> None:
        self.apply_stack()
        self.example.write_text("new base\n", encoding="utf-8")
        self.addon.write_text("addon base\n", encoding="utf-8")

        addon_first = self.run_script(patch=self.addon_patch)
        self.assertNotEqual(addon_first.returncode, 0)
        self.assertIn("apply or recover spec-capture.patch first", addon_first.stderr)
        self.assertEqual(self.sink.read_text(encoding="utf-8"), "new sink\n")

        base_result = self.run_script()
        addon_result = self.run_script(patch=self.addon_patch)
        self.assertEqual(base_result.returncode, 0, base_result.stderr)
        self.assertEqual(addon_result.returncode, 0, addon_result.stderr)
        self.assertEqual(self.example.read_text(encoding="utf-8"), "new patched\n")
        self.assertEqual(self.addon.read_text(encoding="utf-8"), "addon patched\n")
        self.assertEqual(self.sink.read_text(encoding="utf-8"), "new sink\n")
        self.assert_order("spec-capture.patch", "qwen3.5-eagle3.patch")

    def test_migrates_legacy_receipt_and_adopts_directly_applied_addon(self) -> None:
        self.example.write_text("new patched\n", encoding="utf-8")
        self.sink.write_text("new sink\n", encoding="utf-8")
        self.base_record.write_text(
            self.base_patch.read_text(encoding="utf-8"), encoding="utf-8"
        )

        base_result = self.run_script()
        self.assertEqual(base_result.returncode, 0, base_result.stderr)
        self.assertIn("migrated legacy", base_result.stdout)
        self.assert_order("spec-capture.patch")

        self.addon.write_text("addon patched\n", encoding="utf-8")
        addon_result = self.run_script(patch=self.addon_patch)
        self.assertEqual(addon_result.returncode, 0, addon_result.stderr)
        self.assertIn("adopted", addon_result.stdout)
        self.assertTrue(self.addon_record.exists())
        self.assert_order("spec-capture.patch", "qwen3.5-eagle3.patch")

    def test_rejects_removed_v0514_target(self) -> None:
        result = self.run_script("--target", "v0.5.14")

        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported SGLang patch target: v0.5.14", result.stderr)


if __name__ == "__main__":
    unittest.main()
