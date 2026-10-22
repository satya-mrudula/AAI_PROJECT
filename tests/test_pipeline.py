"""
Not aiming for exhaustive coverage here - just the things that would be
genuinely embarrassing to get wrong: gamma actually brightening in the
direction it's supposed to (this broke once already, see git history),
the denoise fallback chain actually engaging when the preferred method
fails, the planner actually falling back to heuristics without an API
key, and the full pipeline running start to finish on a real sample
without throwing.

Run with:  python3 -m unittest discover -s tests -v
"""

import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from agents.denoise_agent import DenoiseAgent
from agents.models import EnhancementPlan
from agents.orchestrator import Pipeline
from agents.planning_agent import PlanningAgent
from agents.review_agent import ReviewAgent
from agents.scene_agent import SceneAgent
from knowledge.retriever import KnowledgeRetriever
from toolbox.image_tools import build_default_registry, load_image
from toolbox.registry import ToolRegistry
from vision.exposure import apply_gamma, brightness_stats


def make_dark_noisy_image(width=200, height=150, mean_level=35, noise_std=20, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), mean_level, dtype=np.float32)
    base += rng.normal(0, noise_std, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)


class GammaCorrectionTests(unittest.TestCase):
    """Regression tests for the direction bug: gamma < 1 must brighten a
    dark image, gamma > 1 must darken it. Everything downstream (the
    heuristic planner, the LLM prompt, the safety preset) assumes this."""

    def setUp(self):
        self.dark_image = make_dark_noisy_image(mean_level=40, noise_std=5)

    def test_gamma_below_one_brightens(self):
        original_mean = brightness_stats(self.dark_image)["mean"]
        brightened = apply_gamma(self.dark_image, 0.5)
        self.assertGreater(brightness_stats(brightened)["mean"], original_mean)

    def test_gamma_above_one_darkens(self):
        original_mean = brightness_stats(self.dark_image)["mean"]
        darkened = apply_gamma(self.dark_image, 1.8)
        self.assertLess(brightness_stats(darkened)["mean"], original_mean)

    def test_gamma_one_is_identity(self):
        unchanged = apply_gamma(self.dark_image, 1.0)
        np.testing.assert_array_equal(unchanged, self.dark_image)


class ToolRegistryTests(unittest.TestCase):
    def test_execute_logs_success(self):
        registry = ToolRegistry()
        registry.register("double", lambda x: x * 2, "doubles a number")
        result = registry.execute("double", 21)
        self.assertEqual(result, 42)
        self.assertTrue(registry.call_log[-1].ok)

    def test_execute_logs_and_reraises_failure(self):
        registry = ToolRegistry()

        def broken(_x):
            raise ValueError("nope")

        registry.register("broken", broken, "always fails")
        with self.assertRaises(ValueError):
            registry.execute("broken", 1)
        self.assertFalse(registry.call_log[-1].ok)
        self.assertIn("nope", registry.call_log[-1].error)

    def test_unknown_tool_raises_keyerror(self):
        registry = ToolRegistry()
        with self.assertRaises(KeyError):
            registry.execute("does_not_exist")


class SceneAgentTests(unittest.TestCase):
    def test_flags_dark_and_noisy_scene(self):
        registry = build_default_registry()
        image = make_dark_noisy_image(mean_level=30, noise_std=35)
        scene = SceneAgent(registry).analyze(image)

        self.assertTrue(scene.is_underexposed)
        self.assertLess(scene.mean_brightness, 60)
        self.assertIn("underexposed", scene.summary)

    def test_does_not_flag_well_lit_scene(self):
        registry = build_default_registry()
        bright_image = np.full((150, 200, 3), 150, dtype=np.uint8)
        scene = SceneAgent(registry).analyze(bright_image)
        self.assertFalse(scene.is_underexposed)


class DenoiseAgentFallbackTests(unittest.TestCase):
    def test_falls_back_when_preferred_method_fails(self):
        registry = build_default_registry()
        registry._tools["denoise_nlmeans"].func = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("simulated failure")
        )

        agent = DenoiseAgent(registry)
        image = make_dark_noisy_image()
        plan = EnhancementPlan(
            gamma=0.8, clahe_clip=2.0, clahe_tile=8, denoise_method="nlmeans",
            denoise_strength=10, sharpen_amount=0.4, reasoning=["test"], source="test",
        )

        result, method_used, fallback_triggered = agent.apply(image, plan)

        self.assertTrue(fallback_triggered)
        self.assertEqual(method_used, "denoise_bilateral")
        self.assertEqual(result.shape, image.shape)

    def test_no_fallback_when_preferred_method_works(self):
        registry = build_default_registry()
        agent = DenoiseAgent(registry)
        image = make_dark_noisy_image()
        plan = EnhancementPlan(
            gamma=0.8, clahe_clip=2.0, clahe_tile=8, denoise_method="nlmeans",
            denoise_strength=10, sharpen_amount=0.4, reasoning=["test"], source="test",
        )

        _, method_used, fallback_triggered = agent.apply(image, plan)
        self.assertFalse(fallback_triggered)
        self.assertEqual(method_used, "denoise_nlmeans")


class PlanningAgentTests(unittest.TestCase):
    def test_uses_heuristic_when_no_api_key_configured(self):
        original_key = config.GEMINI_API_KEY
        config.GEMINI_API_KEY = None
        try:
            planner = PlanningAgent(KnowledgeRetriever())
            self.assertIsNone(planner._client)

            registry = build_default_registry()
            scene = SceneAgent(registry).analyze(make_dark_noisy_image(mean_level=40))
            plan = planner.decide(scene)

            self.assertEqual(plan.source, "heuristic-fallback")
            self.assertEqual(plan.knowledge_used, [])
            self.assertTrue(0.3 <= plan.gamma <= 1.0)
        finally:
            config.GEMINI_API_KEY = original_key

    def test_parses_successful_gemini_plan_and_usage(self):
        class Usage:
            prompt_token_count = 123
            candidates_token_count = 45

        class Response:
            text = '{"gamma":0.75,"clahe_clip":2.2,"clahe_tile":8,"denoise_method":"bilateral","denoise_strength":9,"sharpen_amount":0.4,"reasoning":"balance brightness and noise"}'
            usage_metadata = Usage()

        class Models:
            def generate_content(self, **kwargs):
                self.kwargs = kwargs
                return Response()

        class FakeClient:
            def __init__(self):
                self.models = Models()

        planner = PlanningAgent(KnowledgeRetriever())
        planner._client = FakeClient()
        registry = build_default_registry()
        scene = SceneAgent(registry).analyze(make_dark_noisy_image(mean_level=40))
        plan = planner.decide(scene)

        self.assertEqual(plan.source, "gemini+rag")
        self.assertAlmostEqual(plan.gamma, 0.75)
        self.assertEqual(planner.input_tokens, 123)
        self.assertEqual(planner.output_tokens, 45)

    def test_falls_back_to_heuristic_when_gemini_call_raises(self):
        class BrokenModels:
            def generate_content(self, **kwargs):
                raise ConnectionError("simulated network failure")

        class FakeClient:
            def __init__(self):
                self.models = BrokenModels()

        original_key = config.GEMINI_API_KEY
        config.GEMINI_API_KEY = "fake-key-for-test"
        try:
            planner = PlanningAgent(KnowledgeRetriever())
            planner._client = FakeClient()
            registry = build_default_registry()
            scene = SceneAgent(registry).analyze(make_dark_noisy_image(mean_level=40))
            plan = planner.decide(scene)
            self.assertEqual(plan.source, "heuristic-fallback")
            self.assertIn("ConnectionError", planner.last_llm_error)
        finally:
            config.GEMINI_API_KEY = original_key


class ReviewAgentTests(unittest.TestCase):
    def test_dark_image_scores_worse_than_corrected_one(self):
        review = ReviewAgent()
        dark = make_dark_noisy_image(mean_level=30, noise_std=25)
        corrected = apply_gamma(dark, 0.5)

        dark_report = review.evaluate(dark, dark)
        corrected_report = review.evaluate(dark, corrected)

        self.assertGreater(corrected_report.overall, dark_report.overall)

    def test_feedback_mentions_darkness_for_unfixed_image(self):
        review = ReviewAgent()
        dark = make_dark_noisy_image(mean_level=25, noise_std=5)
        report = review.evaluate(dark, dark)
        self.assertIn("dark", report.feedback)


class FullPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        original_key = config.GEMINI_API_KEY
        config.GEMINI_API_KEY = None
        self.addCleanup(lambda: setattr(config, "GEMINI_API_KEY", original_key))
        self.addCleanup(lambda: shutil.rmtree(self.tmp_dir, ignore_errors=True))

    def test_runs_end_to_end_on_sample_image(self):
        sample_path = ROOT / "samples" / "dim_room.jpg"
        self.assertTrue(sample_path.exists(), "expected samples/dim_room.jpg to exist - run scripts/make_test_image.py")

        output_path = self.tmp_dir / "enhanced.jpg"
        report_path = self.tmp_dir / "report.md"

        pipeline = Pipeline(max_reflection_rounds=1)
        run_report = pipeline.run(str(sample_path), output_path=str(output_path), report_path=str(report_path))

        self.assertTrue(output_path.exists())
        self.assertTrue(report_path.exists())
        self.assertGreaterEqual(len(run_report.attempts), 1)
        self.assertIsInstance(run_report.final_overall_score, float)

        result_image = load_image(str(output_path))
        original_image = load_image(str(sample_path))
        self.assertEqual(result_image.shape, original_image.shape)

    def test_harder_sample_can_trigger_reflection(self):
        sample_path = ROOT / "samples" / "dark_alley.jpg"
        self.assertTrue(sample_path.exists(), "expected samples/dark_alley.jpg to exist - run scripts/make_test_image.py --harsh")

        output_path = self.tmp_dir / "enhanced.jpg"
        pipeline = Pipeline(max_reflection_rounds=2)
        run_report = pipeline.run(str(sample_path), output_path=str(output_path))

        # this scene is deliberately extreme, so we're not asserting it
        # passes the quality bar - just that the loop ran, produced a
        # bounded number of attempts, and left a usable image behind
        self.assertGreaterEqual(len(run_report.attempts), 1)
        self.assertLessEqual(len(run_report.attempts), pipeline.max_reflection_rounds + 2)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
