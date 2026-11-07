import os,sys,unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from timeline import shot_window, master_frame_count, frame_to_sample

class T(unittest.TestCase):
    def test_standard(self):
        F=243
        self.assertEqual(master_frame_count(3,F,1),727)
        a=shot_window(0,F,1); b=shot_window(1,F,1); c=shot_window(2,F,1)
        self.assertEqual((a.generation_start_frame,a.trim_head_frames,a.kept_frames),(0,0,243))
        self.assertEqual((b.generation_start_frame,b.trim_head_frames,b.kept_frames),(242,1,242))
        self.assertEqual((c.generation_start_frame,c.trim_head_frames,c.kept_frames),(484,1,242))
        self.assertEqual(b.kept_start_frame,243)
        self.assertEqual(c.kept_start_frame,485)
    def test_motion22(self):
        F=243
        self.assertEqual(master_frame_count(3,F,22),685)
        b=shot_window(1,F,22)
        self.assertEqual((b.generation_start_frame,b.trim_head_frames,b.kept_start_frame,b.kept_frames),
                         (221,22,243,221))
    def test_samples(self):
        self.assertEqual(frame_to_sample(24,32000,24.0),32000)
if __name__=="__main__":
    unittest.main()
