import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("keyframes_stable", HERE / "keyframes_stable.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ParserTests(unittest.TestCase):
    def test_percent_triplet(self):
        self.assertEqual(m._parse_positions("0%, 50%, 100%", 243), [0, 121, 242])
    def test_absolute(self):
        self.assertEqual(m._parse_positions("0, 121, 242", 243), [0, 121, 242])
    def test_old_fraction(self):
        self.assertEqual(m._parse_positions("0, 0.5, 1", 243), [0, 121, 242])

if __name__ == "__main__":
    unittest.main()
