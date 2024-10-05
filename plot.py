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


class VolCRatePlot(_Plot):

    def __init__(self, figsize=(7, 6), **fig_kw):
        super().__init__(figsize, **fig_kw)

    def plot_single_group(self, dfs, labels):
        for df in dfs:
            pass


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

    def finish(self):
        self._finalize_legend()