# visualize_touch_test.py
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from visualize_touch_temporal import TimeSeriesVideoMaker, df_to_time_and_signals


class TestMakeTimeSeriesVideo(unittest.TestCase):
    def setUp(self):
        self.out_path = Path(__file__).with_name("test_video_unittest.mp4")
        if self.out_path.exists():
            self.out_path.unlink()

    # def tearDown(self):
    #     if self.out_path.exists():
    #         self.out_path.unlink()

    def test_make_timeseries_video(self):
        # ---- 0〜499 frame を 24fps で表現した時間軸 ----
        fps = 24
        n_frames = 500  # frame index: 0..499
        time = np.arange(n_frames) / fps  # 秒単位の time
        # 1) df を読み込む
        csv_file = './animate/gn_outputs_cli_scaffold_bar.csv'
        df = pd.read_csv(csv_file)  # indexにHeadが来る想定

        # 2) DataFrame から time と signals_dict を作成
        time, signals_dict = df_to_time_and_signals(df, original_fps=24.0)
 
        for key in signals_dict.keys():
            signals_dict[key] = signals_dict[key]*(-1000)
 
        # # 3) 表示したい部位を選択
        selected_parts = [
            # "shoulder L",
            # "shoulder R",
            "hand L",
            "hand R",
            "foot L",
            "foot R",
        ]
        colors = {  "hand L":"#A72F2F",
                    "hand R":"#387928",
                    "foot L":"#3842A0",
                    "foot R":"#91A132",
        }

        # selected_parts = None
        self.out_path="timeseries_24fps_hand_foot_scaffold_bar.mp4"
        # 4) 24fps動画を出力
        maker = TimeSeriesVideoMaker(
            fps=24,
            fig_width=16,
            base_fig_height=5,
            height_per_part=1.3,
        )

        maker.make_video(time, signals_dict, color=colors, linewidth=3,ylim=-100,
                 selected_parts=selected_parts, out_path=self.out_path)

        self.out_path = Path(self.out_path)

        self.assertTrue(self.out_path.exists(), "動画ファイルが生成されていません")
        self.assertGreater(self.out_path.stat().st_size, 0, "動画ファイルが空です")


if __name__ == "__main__":
    unittest.main()
