import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class _Plot(object):

    def __init__(self, figsize, **fig_kw):
        self.fig = plt.figure(figsize=figsize, **fig_kw)
        self.ax = plt.axes()
        self.legend_elements = list()

    def _register_legend(self, **line_kwargs):
        line2d = Line2D([0], [0], **line_kwargs)
        self.legend_elements.append(line2d)

    def _finalize_legend(self, **legend_kwargs):
        self.ax.legend(handles=self.legend_elements, **legend_kwargs)

    def finish(self):
        self._finalize_legend()


class VolCRatePlot(_Plot):

    def __init__(self, figsize=(7, 6), **fig_kw):
        super().__init__(figsize, **fig_kw)
        plt.xlabel('C-rate')
        plt.ylabel('Specific Capacity (mAh/g)')

    def plot_single_group(self, data, label=None, **line_kwargs):
        for ser in data:
            plt.plot(ser.index, ser, **line_kwargs)
        if label:
            self._register_legend(label=label, **line_kwargs)

    def plot_combined_df(self, df, label=None, **line_kwargs):
        plt.plot(df, **line_kwargs)
        if label:
            self._register_legend(label=label, **line_kwargs)


class VolCapPlot(_Plot):

    def __init__(self, figsize=(7, 6), **fig_kw):
        super().__init__(figsize, **fig_kw)
        plt.xlabel('Specific Capacity (mAh/g)')
        plt.ylabel('Voltage (V)')

    def plot_single_group(self, data, label=None, **line_kwargs):
        # Require Voltage and Specific_Capacity columns
        for df in data:
            plt.plot(df['Specific_Capacity'], df['Voltage'], **line_kwargs)
        if label:
            self._register_legend(label=label, **line_kwargs)


class CapRetentionPlot(_Plot):

    def __init__(self, figsize=(7, 6), **fig_kw):
        super().__init__(figsize, **fig_kw)
        plt.xlabel('Cycle Counts')
        plt.xlim(0, None)
        plt.ylabel('Capacity Retention (%)')

    def plot_single_group(self, arrs, label, **line_kwargs):
        for arr in arrs:
            cycle_nums = np.arange(1, len(arr) + 1, 1)
            plt.plot(cycle_nums, arr, **line_kwargs)
        self._register_legend(label=label, **line_kwargs)


class NyquistPlot(_Plot):

    def __init__(self, figsize=(10, 5), **fig_kw):
        super().__init__(figsize, **fig_kw)

    def add_zoom_axes(self, zoom_x=(0.1, 0.3), zoom_y=(-0.1, 0.1), axes_position=(0.08, 0.65, 0.15, 0.3)):
        axins = self.ax.inset_axes(
            axes_position,
            xlim=zoom_x, ylim=zoom_y)
        axins.grid()
        axins.set_aspect(1)