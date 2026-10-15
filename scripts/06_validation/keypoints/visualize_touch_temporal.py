import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import pandas as pd

def df_to_time_and_signals(df: pd.DataFrame, original_fps: float):
    """
    df : DataFrame with a "frame" column and one column per body part
    original_fps : fps of the original data (e.g. 120.0)
    """
    # Time axis (seconds): frame / original_fps
    frames = df["frame"].to_numpy()
    time = frames / original_fps  # e.g. frame 0 -> 0s, frame 120 -> 1s

    # Build signals_dict (treat every column except "frame" as a body part)
    signals_dict = {
        col: df[col].to_numpy()
        for col in df.columns
        if col != "frame"
    }

    return time, signals_dict


class TimeSeriesVideoMaker:
    """
    Class that creates a video from time-series data.

    - If selected_parts has 1 entry, use the original single-plot logic almost as is
    - If selected_parts has 2 or more entries, draw one subplot per body part, stacked vertically
    - Figure size can be adjusted with fig_width, base_fig_height, height_per_part
    """

    def __init__(
        self,
        fps=24,
        fig_width=8,
        base_fig_height=4,    # height for single-plot mode
        height_per_part=2.5,  # height per subplot in multi-plot mode
        dpi=150,
    ):
        self.fps = fps
        self.fig_width = fig_width
        self.base_fig_height = base_fig_height
        self.height_per_part = height_per_part
        self.dpi = dpi

    # ---------- Common resampling ----------
    def _set_ylim(self, ylim ,ax,resampled):
        if ylim is not None:
            ax.set_ylim(ylim,0)
        else:
            ax.set_ylim(resampled.max(),0)
        
    def _resample(self, time, signals_dict, selected_parts):
        t_start = float(time[0])
        t_end = float(time[-1])
        duration = t_end - t_start

        # Number of frames when sampling duration seconds at fps
        n_frames = max(1, int(duration * self.fps))

        new_t = np.linspace(t_start, t_end, n_frames)

        # Linearly interpolate each body part onto new_t
        resampled = {}
        for part in selected_parts:
            resampled[part] = np.interp(new_t, time, signals_dict[part])

        return new_t, resampled

    def _get_color_for_part(self, color, part):
        """
        color can be:
        - None:  default color
        - str:   same color for all body parts
        - dict:  {"part_name": "color"}, color per body part
        """
        if color is None:
            return None
        if isinstance(color, dict):
            return color.get(part, None)
        return color

    # ---------- Main entry point (called externally) ----------

    def make_video(
        self,
        time,                   # 1D array (seconds)
        signals_dict,           # {"part_name": 1D array}
        color=None,
        linewidth=1,
        ylim = -100,
        selected_parts=None,    # list of body parts to show; all if None
        out_path="output.mp4",
    ):
        if selected_parts is None:
            selected_parts = list(signals_dict.keys())

        if len(selected_parts) == 0:
            raise ValueError("selected_parts is empty.")

        # Use the original single-plot logic for a single body part
        if len(selected_parts) == 1:
            self._make_single(time, signals_dict, color,linewidth, ylim, selected_parts[0], out_path)
        # Split into subplots for multiple body parts
        else:
            self._make_multi(time, signals_dict, color,linewidth, ylim, selected_parts, out_path)

    # ---------- Single-plot version based on the original code ----------

    def _make_single(self, time, signals_dict, color,linewidth, ylim, part, out_path):
        # Mostly reuses the original function logic for a single body part
        t_start = float(time[0])
        t_end = float(time[-1])
        duration = t_end - t_start

        n_frames = max(1, int(duration * self.fps))
        new_t = np.linspace(t_start, t_end, n_frames)

        # Interpolate only one body part
        resampled = np.interp(new_t, time, signals_dict[part])

        # ---- Plot setup ----
        fig, ax = plt.subplots(figsize=(self.fig_width, self.base_fig_height))

        line_color = self._get_color_for_part(color, part)
        plot_kwargs = {}
        if line_color is not None:
            plot_kwargs["color"] = line_color

        line, = ax.plot(new_t, resampled, label=part,linewidth=linewidth, **plot_kwargs)

        # Vertical line indicating the playback position
        cursor = ax.axvline(new_t[0], linestyle="--", linewidth=linewidth)

        ax.set_xlim(0, new_t[-1])
        self._set_ylim(ylim, ax, resampled)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Distance [mm]")
        ax.legend(loc="upper right")

        # ---- Save as video ----
        writer = FFMpegWriter(fps=self.fps)

        with writer.saving(fig, out_path, dpi=self.dpi):
            for t_cur in new_t:
                cursor.set_xdata(t_cur)
                ax.set_title(f"{part}  t = {t_cur:.3f} s")
                writer.grab_frame()

        plt.close(fig)

    # ---------- Multi-part version: split into subplots ----------

    def _make_multi(self, time, signals_dict, color, linewidth, ylim, selected_parts, out_path):
        # Use the common resampling logic
        new_t, resampled = self._resample(time, signals_dict, selected_parts)

        n_parts = len(selected_parts)
        fig_height = self.height_per_part * n_parts

        # Stack one subplot per body part vertically
        fig, axes = plt.subplots(
            n_parts, 1, sharex=True,
            figsize=(self.fig_width, fig_height)
        )
        fig.supylabel('Distance [mm]')
        # Assumes 2 or more body parts, but just in case
        if n_parts == 1:
            axes = [axes]

        cursors = {}

        # Draw each body part in its own subplot
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
                # Update the vertical lines in all subplots
                for part in selected_parts:
                    cursors[part].set_xdata(t_cur)

                fig.suptitle(f"t = {t_cur:.3f} s")
                writer.grab_frame()

        plt.close(fig)



def main():
    
    # ==== Test with frames 0-499 at 24fps ====
    fps = 24
    n_frames = 500                # 500 frames, 0-499
    frame_idx = np.arange(n_frames)

    # Time axis (for checking)
    t = frame_idx / fps           # 0〜 (n_frames-1)/fps [s]

    # Create two arbitrary dummy signals
    signals_dict = {
        "part_A": np.sin(2 * np.pi * 0.5 * t),         # 0.5Hz sine wave
        "part_B": 0.5 * np.cos(2 * np.pi * 1.0 * t),   # 1Hz cosine wave
    }
    
    maker = TimeSeriesVideoMaker(
    fps=24,
    fig_width=8,
    base_fig_height=4,
    height_per_part=2.5,
    )

    # Single body part only
    maker.make_video(t, signals_dict, color="C0",
                 selected_parts=["part_A"], out_path="single_part.mp4")
    

if __name__ == '__main__':
    main()