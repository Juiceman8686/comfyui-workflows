import contextlib
import importlib.util
import io
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2_nodes", HERE / "v2_nodes.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# the v2.1 realism schedule for shift 12 (beta 20 + extend below 0.7), 24 steps
V21 = [1.000, 0.999, 0.997, 0.993, 0.988, 0.983, 0.975, 0.966, 0.955, 0.941, 0.923, 0.901,
       0.872, 0.835, 0.787, 0.721, 0.629, 0.567, 0.504, 0.421, 0.339, 0.242, 0.146, 0.073, 0.0]
TURBO8 = [1.0, 0.988, 0.973, 0.952, 0.923, 0.878, 0.800, 0.632, 0.0]
TURBO4 = [1.0, 0.973, 0.923, 0.800, 0.0]
FACE_REFINE = [0.889, 0.866, 0.837, 0.800, 0.750, 0.679, 0.571, 0.387, 0.0]


class _Adapter:
    multiplier = 1.0


class _Hook:
    def __init__(self, strength):
        self.adapter = _Adapter()
        self.adapter.multiplier = strength
        self.multiplier = strength


class _T:
    """Stands in for a tensor: .flatten().tolist() and [0].item()."""
    def __init__(self, *v):
        self.v = list(v)
    def flatten(self):
        return self
    def tolist(self):
        return list(self.v)
    def __getitem__(self, i):
        return self.v[i]


class WindowFactorTests(unittest.TestCase):
    def test_default_window_is_always_on(self):
        for s in (1.0, 0.75, 0.3, 0.0):
            self.assertEqual(m._window_factor(s, 1.0), 1.0)

    def test_window_edges(self):
        self.assertEqual(m._window_factor(0.90, 0.75), 0.0)
        self.assertEqual(m._window_factor(0.75, 0.75), 0.0)
        self.assertAlmostEqual(m._window_factor(0.70, 0.75), 0.5)
        self.assertEqual(m._window_factor(0.65, 0.75), 1.0)
        self.assertEqual(m._window_factor(0.05, 0.75), 1.0)
        self.assertEqual(m._window_factor(0.0, 0.75), 1.0)

    def test_sigma_of(self):
        self.assertEqual(m._sigma_of(_T(0.42)), 0.42)
        self.assertEqual(m._sigma_of(0.42), 0.42)


class StepTargetTests(unittest.TestCase):
    def test_exact_schedule_entries(self):
        self.assertEqual(m._step_target(1.000, V21), (0, 0.999))
        self.assertEqual(m._step_target(0.787, V21), (14, 0.721))
        self.assertEqual(m._step_target(0.073, V21), (23, 0.0))

    def test_inside_a_step(self):
        # a sampler evaluating between two entries belongs to the step that started above it
        self.assertEqual(m._step_target(0.70, V21), (15, 0.629))

    def test_no_schedule_falls_back_to_own_sigma(self):
        self.assertEqual(m._step_target(0.70, None), (None, 0.70))
        self.assertEqual(m._schedule_of(None), None)
        self.assertEqual(m._schedule_of({}), None)
        self.assertEqual(m._schedule_of({"sample_sigmas": _T(*TURBO4)}), TURBO4)

    def test_transformer_options_arrive_positionally(self):
        # BaseModel.apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        # hands transformer_options to an APPLY_MODEL wrapper as args[3]; a keyword is only a fallback
        to = {"sample_sigmas": _T(*TURBO4)}
        self.assertIs(m._transformer_options_of((None, None, None, to), {}), to)
        self.assertIs(m._transformer_options_of((), {"transformer_options": to}), to)
        self.assertEqual(m._transformer_options_of((), {}), None)
        self.assertEqual(m._transformer_options_of((None, None, None, "not a dict"), {}), None)

    def test_shipped_schedules_at_075(self):
        self.assertEqual(m._window_summary(V21, 0.75), (9, 1, 14))      # full, fading, off
        self.assertEqual(m._window_summary(TURBO8, 0.75), (2, 0, 6))
        self.assertEqual(m._window_summary(TURBO4, 0.75), (1, 0, 3))    # the last step keeps the LoRA
        self.assertEqual(m._window_summary(FACE_REFINE, 0.75), (3, 1, 4))
        self.assertEqual(m._window_summary(V21, 1.0), (24, 0, 0))


class WrapperTests(unittest.TestCase):
    def test_wrapper_uses_the_noise_a_step_produces(self):
        realism, lms = _Hook(0.65), _Hook(0.6)
        seen = []
        def executor(x, t, *a, **k):
            seen.append((round(realism.adapter.multiplier, 4), round(lms.adapter.multiplier, 4), round(realism.multiplier, 4)))
            return "out"
        w = m._look_window_wrapper([(realism, 0.65), (lms, 0.6)], 0.75)
        to = {"sample_sigmas": _T(*V21)}
        # called exactly as BaseModel.apply_model does: (x, t, c_concat, c_crossattn, control, transformer_options)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(w(executor, "x", _T(1.000), None, None, None, to), "out")   # step 1 -> 0.999: off
            self.assertEqual(w(executor, "x", _T(1.000), None, None, None, to), "out")   # uncond batch, same step
            self.assertEqual(w(executor, "x", _T(0.835), None, None, None, to), "out")   # step 14 -> 0.787: off
            self.assertEqual(w(executor, "x", _T(0.787), None, None, None, to), "out")   # step 15 -> 0.721: fading
            self.assertEqual(w(executor, "x", _T(0.721), None, None, None, to), "out")   # step 16 -> 0.629: full
            self.assertEqual(w(executor, "x", _T(0.073), None, None, None, to), "out")   # last step -> 0: full
        self.assertEqual(seen[0], (0.0, 0.0, 0.0))
        self.assertEqual(seen[2], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(seen[3][0], 0.65 * 0.29, places=3)
        self.assertAlmostEqual(seen[3][1], 0.6 * 0.29, places=3)
        self.assertEqual(seen[4], (0.65, 0.6, 0.65))
        self.assertEqual(seen[5], (0.65, 0.6, 0.65))
        printed = out.getvalue()
        self.assertEqual(printed.count("look window 0.75: 24 steps"), 1)   # once per run, not per batch
        self.assertIn("run on the last 9 (+1 fading in) and are off on the first 14", printed)

    def test_turbo_final_step_keeps_the_lora(self):
        realism = _Hook(0.65)
        mults = []
        def executor(x, t, *a, **k):
            mults.append(realism.adapter.multiplier)
        w = m._look_window_wrapper([(realism, 0.65)], 0.75)
        to = {"sample_sigmas": _T(*TURBO4)}
        with contextlib.redirect_stdout(io.StringIO()):
            for s in TURBO4[:-1]:
                w(executor, "x", _T(s), None, None, None, to)
        self.assertEqual(mults, [0.0, 0.0, 0.0, 0.65])

    def test_keyword_transformer_options_also_work(self):
        realism = _Hook(0.65)
        mults = []
        def executor(x, t, *a, **k):
            mults.append(realism.adapter.multiplier)
        w = m._look_window_wrapper([(realism, 0.65)], 0.75)
        with contextlib.redirect_stdout(io.StringIO()):
            w(executor, "x", _T(0.800), transformer_options={"sample_sigmas": _T(*TURBO4)})
        self.assertEqual(mults, [0.65])

    def test_minimum_value_still_gives_a_full_strength_final_step(self):
        self.assertEqual(m._window_factor(0.0, m._LOOK_LORA_MIN), 1.0)
        self.assertEqual(m._window_summary(TURBO4, m._LOOK_LORA_MIN), (1, 0, 3))

    def test_without_schedule_uses_own_sigma(self):
        realism = _Hook(0.65)
        mults = []
        def executor(x, t, *a, **k):
            mults.append(realism.adapter.multiplier)
        w = m._look_window_wrapper([(realism, 0.65)], 0.75)
        with contextlib.redirect_stdout(io.StringIO()):
            w(executor, "x", _T(0.9)); w(executor, "x", _T(0.5))
        self.assertEqual(mults, [0.0, 0.65])


class ApplyPlumbingTests(unittest.TestCase):
    """H3RDLoraStack.apply passes the right items and window to the runtime path and reports it."""

    def setUp(self):
        self.calls = []
        self._saved = (m._find_lora, m._runtime_lora_support, m._apply_loras_runtime,
                       m._earlier_runtime_loras, m._apply_lora, m._look_window_support)
        m._find_lora = lambda exact, pats: exact[0]
        m._runtime_lora_support = lambda: True
        m._look_window_support = lambda: True
        m._earlier_runtime_loras = lambda model: False
        m._apply_lora = lambda model, name, strength: model
        self.hooked = 10

        def fake_runtime(model, items, look_below=1.0):
            self.calls.append((items, look_below))
            windowed_any = any(w for _n, _s, w in items)
            return model, [(self.hooked, 0, 0)] * len(items), (look_below < 1.0 and windowed_any and self.hooked > 0)
        m._apply_loras_runtime = fake_runtime

    def tearDown(self):
        (m._find_lora, m._runtime_lora_support, m._apply_loras_runtime,
         m._earlier_runtime_loras, m._apply_lora, m._look_window_support) = self._saved

    def _run(self, **kw):
        node = m.H3RDLoraStack()
        with contextlib.redirect_stdout(io.StringIO()):
            return node.apply("MODEL", realism_people=True, realism_people_strength=0.65,
                              motion_continuity=True, motion_continuity_strength=0.9,
                              custom_lora_1="mine.safetensors", custom_lora_1_strength=0.5, **kw)

    def test_windowed_flags_and_report(self):
        _model, prefix, report = self._run(look_loras_below_noise=0.75)
        items, look_below = self.calls[0]
        self.assertEqual(look_below, 0.75)
        flags = {name: w for name, _s, w in items}
        self.assertTrue(flags["h3-realism-people-t2v-i2v-r2v.safetensors"])
        self.assertFalse(flags["Motion_Repair_V2.safetensors"])
        self.assertTrue(flags["mine.safetensors"])
        self.assertIn("look window: the look LoRAs are OFF on the steps that produce noise above 0.75", report)
        self.assertIn("full strength on the steps below noise 0.65", report)
        self.assertIn("fal Realism People: ON  strength 0.65", report)
        self.assertIn(", look window]", report)
        self.assertIn("r34l1sm", prefix)

    def test_default_keeps_v21_behaviour(self):
        _model, _prefix, report = self._run()
        self.assertEqual(self.calls[0][1], 1.0)
        self.assertNotIn("look window", report)

    def test_value_is_clamped_to_minimum(self):
        self._run(look_loras_below_noise=0.0)
        self.assertEqual(self.calls[0][1], m._LOOK_LORA_MIN)

    def test_merged_mode_reports_ignored_window(self):
        _model, _prefix, report = self._run(lora_apply_mode="merged", look_loras_below_noise=0.75)
        self.assertEqual(self.calls, [])
        self.assertIn("ignored (merged)", report)

    def test_report_reason_when_no_layers_hooked(self):
        self.hooked = 0
        _model, _prefix, report = self._run(look_loras_below_noise=0.75)
        self.assertIn("no runtime layers", report)
        self.assertNotIn("newer ComfyUI core", report)

    def test_report_reason_when_core_too_old(self):
        m._look_window_support = lambda: False
        self.hooked = 0
        _model, _prefix, report = self._run(look_loras_below_noise=0.75)
        self.assertIn("newer ComfyUI core", report)

    def test_old_saved_value_order_still_works(self):
        _model, _prefix, report = self._run(lora_apply_mode="runtime")
        self.assertNotIn("look window", report)


if __name__ == "__main__":
    unittest.main()
