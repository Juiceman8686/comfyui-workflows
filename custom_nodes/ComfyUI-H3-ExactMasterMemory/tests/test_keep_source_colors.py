import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v2_nodes", HERE / "v2_nodes.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ColourMathTests(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        rgb = rng.random((4, 5, 3))
        y, cb, cr = m._rgb_to_ycc(rgb[..., 0], rgb[..., 1], rgb[..., 2])
        r, g, b = m._ycc_to_rgb(y, cb, cr)
        np.testing.assert_allclose(np.stack((r, g, b), -1), rgb, atol=1e-9)

    def test_grey_has_no_chroma_and_luma_is_bt709(self):
        y, cb, cr = m._rgb_to_ycc(np.array(0.4), np.array(0.4), np.array(0.4))
        self.assertAlmostEqual(float(y), 0.4); self.assertAlmostEqual(float(cb), 0.0); self.assertAlmostEqual(float(cr), 0.0)

    def test_green_strand_takes_the_source_colour_but_keeps_its_brightness(self):
        # upscaled frame: a teal strand (the defect) on dark hair; source: the same strand, dark brown
        target = np.full((1, 4, 4, 3), 0.12); target[0, 1, :, :] = (53 / 255, 81 / 255, 70 / 255)
        source = np.full((1, 4, 4, 3), 0.12); source[0, 1, :, :] = (0.30, 0.24, 0.20)
        r, g, b = m._lock_colours(target, source, chroma=1.0)
        out = np.stack((r, g, b), -1)
        ty = m._rgb_to_ycc(target[..., 0], target[..., 1], target[..., 2])[0]
        oy, ocb, ocr = m._rgb_to_ycc(out[..., 0], out[..., 1], out[..., 2])
        _sy, scb, scr = m._rgb_to_ycc(source[..., 0], source[..., 1], source[..., 2])
        np.testing.assert_allclose(oy, ty, atol=1e-9)            # brightness (detail) of the upscale kept
        np.testing.assert_allclose(ocb, scb, atol=1e-9)          # colour of the normal video
        np.testing.assert_allclose(ocr, scr, atol=1e-9)
        self.assertGreater(out[0, 1, 0, 0], out[0, 1, 0, 1])     # the strand is now reddish-brown, not green
        self.assertGreater(target[0, 1, 0, 1], target[0, 1, 0, 0])

    def test_chroma_zero_is_identity_and_half_is_halfway(self):
        rng = np.random.default_rng(1)
        target, source = rng.random((2, 3, 3, 3)), rng.random((2, 3, 3, 3))
        out0 = np.stack(m._lock_colours(target, source, chroma=0.0), -1)
        np.testing.assert_allclose(out0, target, atol=1e-9)
        out5 = np.stack(m._lock_colours(target, source, chroma=0.5), -1)
        _y, cb5, _cr = m._rgb_to_ycc(out5[..., 0], out5[..., 1], out5[..., 2])
        _y, cbt, _cr = m._rgb_to_ycc(target[..., 0], target[..., 1], target[..., 2])
        _y, cbs, _cr = m._rgb_to_ycc(source[..., 0], source[..., 1], source[..., 2])
        np.testing.assert_allclose(cb5, (cbt + cbs) / 2, atol=1e-9)

    def test_brightness_lock_moves_only_the_coarse_part(self):
        target = np.full((1, 2, 2, 3), 0.6); source = np.full((1, 2, 2, 3), 0.3)
        blur = lambda x: np.full_like(x, x.mean())                  # a stand-in for the Gaussian blur
        r, g, b = m._lock_colours(target, source, chroma=0.0, brightness=1.0, blur=blur)
        np.testing.assert_allclose(np.stack((r, g, b), -1), 0.3, atol=1e-9)
        r, g, b = m._lock_colours(target, source, chroma=0.0, brightness=0.5, blur=blur)
        np.testing.assert_allclose(np.stack((r, g, b), -1), 0.45, atol=1e-9)

    def test_max_shift_keeps_small_nuances_and_cuts_large_ones(self):
        source = np.full((1, 1, 2, 3), 0.3)
        target = source.copy()
        target[0, 0, 0] = (0.31, 0.30, 0.29)          # a nuance: tiny warm shift (< 4/255 in chroma)
        target[0, 0, 1] = (53 / 255, 81 / 255, 70 / 255)   # a strong teal recolouring
        out = np.stack(m._lock_colours(target, source, chroma=1.0, max_shift=4 / 255), -1)
        _y, cb_t, cr_t = m._rgb_to_ycc(target[..., 0], target[..., 1], target[..., 2])
        _y, cb_o, cr_o = m._rgb_to_ycc(out[..., 0], out[..., 1], out[..., 2])
        _y, cb_s, cr_s = m._rgb_to_ycc(source[..., 0], source[..., 1], source[..., 2])
        np.testing.assert_allclose(cb_o[0, 0, 0], cb_t[0, 0, 0], atol=1e-9)      # nuance kept as is
        np.testing.assert_allclose(cr_o[0, 0, 0], cr_t[0, 0, 0], atol=1e-9)
        self.assertLessEqual(abs(cb_o[0, 0, 1] - cb_s[0, 0, 1]), 4 / 255 + 1e-9)   # strong shift cut to 4/255
        self.assertLessEqual(abs(cr_o[0, 0, 1] - cr_s[0, 0, 1]), 4 / 255 + 1e-9)
        self.assertGreater(abs(cr_t[0, 0, 1] - cr_s[0, 0, 1]), 8 / 255)         # (the teal shift is mostly in Cr: ~13/255)

    def test_node_is_registered(self):
        self.assertIn("H3RDKeepSourceColors", m.NODE_CLASS_MAPPINGS)
        inputs = m.H3RDKeepSourceColors.INPUT_TYPES()["required"]
        self.assertEqual(list(inputs), ["images", "source_images", "chroma_from_source", "max_kept_shift",
                                        "brightness_from_source", "coarse_px", "frames_per_batch"])


if __name__ == "__main__":
    unittest.main()
