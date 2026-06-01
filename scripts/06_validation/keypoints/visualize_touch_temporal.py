import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import pandas as pd

def df_to_time_and_signals(df: pd.DataFrame, original_fps: float):
    """
    df : 質問に貼ってくれたような DataFrame
    original_fps : 元データのfps（例：120.0）
    """
    # 時間軸（秒）: frame / original_fps
    frames = df["frame"].to_numpy()
    time = frames / original_fps  # 例: frame 0 → 0s, frame 120 → 1s

    # signals_dict を作成（frame列以外を全部「部位」とみなす）
    signals_dict = {
        col: df[col].to_numpy()
        for col in df.columns
        if col != "frame"
    }

    return time, signals_dict


class TimeSeriesVideoMaker:
    """
    時系列データから動画を作るクラス。

    - selected_parts が 1 個なら元の単一プロット版ロジックをほぼそのまま使用
    - selected_parts が 2 個以上なら，部位ごとにサブプロットを縦に並べて描画
    - figure のサイズは fig_width, base_fig_height, height_per_part で調整可能
    """

    def __init__(
        self,
        fps=24,
        fig_width=8,
        base_fig_height=4,    # single-plot 時の高さ
        height_per_part=2.5,  # multi-plot 時の 1 subplot あたりの高さ
        dpi=150,
    ):
        self.fps = fps
        self.fig_width = fig_width
        self.base_fig_height = base_fig_height
        self.height_per_part = height_per_part
        self.dpi = dpi

    # ---------- 共通のリサンプリング部分 ----------
    def _set_ylim(self, ylim ,ax,resampled):
        if ylim is not None:
            ax.set_ylim(ylim,0)
        else:
            ax.set_ylim(resampled.max(),0)
        
    def _resample(self, time, signals_dict, selected_parts):
        t_start = float(time[0])
        t_end = float(time[-1])
        duration = t_end - t_start

        # duration秒をfpsでサンプリングしたフレーム数
        n_frames = max(1, int(duration * self.fps))

        new_t = np.linspace(t_start, t_end, n_frames)

        # 各部位を new_t に線形補間
        resampled = {}
        for part in selected_parts:
            resampled[part] = np.interp(new_t, time, signals_dict[part])

        return new_t, resampled

    def _get_color_for_part(self, color, part):
        """
        color が:
        - None:  デフォルト色
        - str:   全部位同じ色
        - dict:  {"部位名": "色"} のとき，部位ごとの色指定
        """
        if color is None:
            return None
        if isinstance(color, dict):
            return color.get(part, None)
        return color

    # ---------- 外から呼ぶメイン関数 ----------

    def make_video(
        self,
        time,                   # 1D array（秒）
        signals_dict,           # {"部位名": 1D array}
        color=None,
        linewidth=1,
        ylim = -100,
        selected_parts=None,    # 表示したい部位のリスト。Noneなら全部
        out_path="output.mp4",
    ):
        if selected_parts is None:
            selected_parts = list(signals_dict.keys())

        if len(selected_parts) == 0:
            raise ValueError("selected_parts が空です。")

        # 部位が 1 つなら元の単一プロット版ロジックを使う
        if len(selected_parts) == 1:
            self._make_single(time, signals_dict, color,linewidth, ylim, selected_parts[0], out_path)
        # 複数ならサブプロットで分割
        else:
            self._make_multi(time, signals_dict, color,linewidth, ylim, selected_parts, out_path)

    # ---------- 元コードをベースにした単一プロット版 ----------

    def _make_single(self, time, signals_dict, color,linewidth, ylim, part, out_path):
        # ここはほぼ元の関数のロジックを 1 部位用に再利用
        t_start = float(time[0])
        t_end = float(time[-1])
        duration = t_end - t_start

        n_frames = max(1, int(duration * self.fps))
        new_t = np.linspace(t_start, t_end, n_frames)

        # 1 部位だけ補間
        resampled = np.interp(new_t, time, signals_dict[part])

        # ---- 描画セットアップ ----
        fig, ax = plt.subplots(figsize=(self.fig_width, self.base_fig_height))

        line_color = self._get_color_for_part(color, part)
        plot_kwargs = {}
        if line_color is not None:
            plot_kwargs["color"] = line_color

        line, = ax.plot(new_t, resampled, label=part,linewidth=linewidth, **plot_kwargs)

        # 再生位置を示す縦線
        cursor = ax.axvline(new_t[0], linestyle="--", linewidth=linewidth)

        ax.set_xlim(0, new_t[-1])
        self._set_ylim(ylim, ax, resampled)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [mm]")
        ax.legend(loc="upper right")

        # ---- 動画として保存 ----
        writer = FFMpegWriter(fps=self.fps)

        with writer.saving(fig, out_path, dpi=self.dpi):
            for t_cur in new_t:
                cursor.set_xdata(t_cur)
                ax.set_title(f"{part}  t = {t_cur:.3f} s")
                writer.grab_frame()

        plt.close(fig)

    # ---------- 複数部位用：サブプロット分割版 ----------

    def _make_multi(self, time, signals_dict, color, linewidth, ylim, selected_parts, out_path):
        # 共通のリサンプリングロジックを使用
        new_t, resampled = self._resample(time, signals_dict, selected_parts)

        n_parts = len(selected_parts)
        fig_height = self.height_per_part * n_parts

        # 部位の数だけ縦にサブプロットを並べる
        fig, axes = plt.subplots(
            n_parts, 1, sharex=True,
            figsize=(self.fig_width, fig_height)
        )
        fig.supylabel('Distance [mm]')
        # 部位が 2 個以上の前提だが、念のため
        if n_parts == 1:
            axes = [axes]

        cursors = {}

        # 各サブプロットごとにそれぞれの部位を描画
        for ax, part in zip(axes, selected_parts):
            vals = resampled[part]
            line_color = self._get_color_for_part(color, part)
            plot_kwargs = {}
            if line_color is not None:
                plot_kwargs["color"] = line_color

            ax.plot(new_t, vals, label=part,linewidth=linewidth, **plot_kwargs)
            cursor = ax.axvline(new_t[0], linestyle="--")
            cursors[part] = cursor

            ax.set_xlim(new_t[0], new_t[-1])
            self._set_ylim(ylim, ax, vals)

            ax.set_ylabel(f"{part}")
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")

        writer = FFMpegWriter(fps=self.fps)

        with writer.saving(fig, out_path, dpi=self.dpi):
            for t_cur in new_t:
                # 全サブプロットの縦線を更新
                for part in selected_parts:
                    cursors[part].set_xdata(t_cur)

                fig.suptitle(f"t = {t_cur:.3f} s")
                writer.grab_frame()

        plt.close(fig)



def main():
    
    # ==== 0〜499 frame を 24fps でテスト ====
    fps = 24
    n_frames = 500                # 0〜499 の 500フレーム
    frame_idx = np.arange(n_frames)

    # 時間軸（確認用）
    t = frame_idx / fps           # 0〜 (n_frames-1)/fps [s]

    # ダミー信号を適当に2つ作成
    signals_dict = {
        "part_A": np.sin(2 * np.pi * 0.5 * t),         # 0.5Hz のサイン波
        "part_B": 0.5 * np.cos(2 * np.pi * 1.0 * t),   # 1Hz のコサイン波
    }
    
    maker = TimeSeriesVideoMaker(
    fps=24,
    fig_width=8,
    base_fig_height=4,
    height_per_part=2.5,
    )

    # 1部位だけ
    maker.make_video(t, signals_dict, color="C0",
                 selected_parts=["part_A"], out_path="single_part.mp4")
    

if __name__ == '__main__':
    main()