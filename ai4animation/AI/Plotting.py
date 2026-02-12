import warnings

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def PCA2D(ax, values, title):  # values is batch x seq x dim
    x = ToNumpy(values)

    batch = x.shape[0]
    dim = x.shape[1]

    ax.cla()

    point_alpha = 0.3
    line_alpha = 0.2

    y = PCA(n_components=2).fit_transform(x.reshape(-1, dim)).reshape(batch, 2)
    for i in range(y.shape[0]):
        px = y[i, 0]
        py = y[i, 1]
        ax.plot(px, py, c=(0, 0, 0), alpha=line_alpha)
        ax.scatter(px, py, alpha=point_alpha)

    ax.set_axis_off()
    ax.set_title(title)


def PCA2DSequence(ax, values, title):  # values is batch x seq x dim
    x = ToNumpy(values)

    batch = x.shape[0]
    seq = x.shape[1]
    dim = x.shape[2]

    ax.cla()

    point_alpha = 0.3
    line_alpha = 0.2

    y = PCA(n_components=2).fit_transform(x.reshape(-1, dim)).reshape(batch, seq, 2)
    for i in range(y.shape[0]):
        px = y[i, :, 0]
        py = y[i, :, 1]
        ax.plot(px, py, c=(0, 0, 0), alpha=line_alpha)
        ax.scatter(px, py, alpha=point_alpha)

    ax.set_axis_off()
    ax.set_title(title)


def PlotFunctions(ax, values, title, step=1, yLimits=None):
    values = ToNumpy(values)

    ax.cla()
    for i in range(0, len(values), step):
        ax.plot(values[i])
    if yLimits is not None:
        ax.set_ylim([yLimits[0], yLimits[1]])
    ax.set_axis_off()
    ax.set_title(title)


def PlotGridMap(ax, values, title):
    ax.cla()
    ax.imshow(values, cmap=plt.cm.bone, aspect="auto")
    ax.set_axis_off()
    ax.set_title(title)


def ToNumpy(value):
    return value.detach().cpu().numpy()


class LossHistory:
    def __init__(
        self,
        title,
        ax=None,
        min=None,
        max=None,
        maxHistory=0,
        cumulativeHorizon=100,
        drawInterval=100,
        drawWait=1e-3,
        yScale="linear",
    ):
        plt.ion()
        warnings.filterwarnings(
            "ignore",
            message="Attempt to set non-positive ylim on a log-scaled axis will be ignored.",
        )
        if ax is None:
            _, self.ax = plt.subplots(figsize=(10, 5))
        else:
            self.ax = ax
        self.Title = title
        self.MaxHistory = maxHistory
        self.CumulativeHorizon = cumulativeHorizon
        self.DrawInterval = drawInterval
        self.DrawWait = drawWait
        self.YMin = min
        self.YMax = max
        self.YRange = [0 if min == None else min, 0 if max == None else max]
        self.Functions = {}  # string->[History, Horizon]
        self.Counter = 0
        self.YScale = yScale

    def Close(self):
        plt.close()

    def Add(self, *args):  # arg->(value, label)
        for arg in args:
            value = arg[0]
            label = arg[1]
            if label not in self.Functions:
                self.Functions[label] = ([], [])  # (Value, Cumulative)
            function = self.Functions[label]
            function[0].append(value)
            cumulative = sum(function[0][-self.CumulativeHorizon :]) / len(
                function[0][-self.CumulativeHorizon :]
            )
            function[1].append(cumulative)

            if self.MaxHistory != 0:
                while len(function[0]) > self.MaxHistory:
                    function[0].pop(0)
                while len(function[1]) > self.MaxHistory:
                    function[1].pop(0)

            self.YRange[0] = (
                min(self.YRange[0], 0.5 * cumulative)
                if self.YMin == None
                else self.YRange[0]
            )
            self.YRange[1] = (
                max(self.YRange[1], 2 * cumulative)
                if self.YMax == None
                else self.YRange[1]
            )

        self.Counter += 1
        if self.Counter >= self.DrawInterval:
            self.Counter = 0
            self.Draw()

    def Draw(self):
        self.ax.cla()
        self.ax.set_title(self.Title)
        for label in self.Functions.keys():
            function = self.Functions[label]
            step = max(int(len(function[0]) / self.DrawInterval), 1)
            self.ax.plot(
                function[0][::step],
                label=label + " (" + str(round(self.CumulativeValue(label), 3)) + ")",
            )
            self.ax.plot(function[1][::step], c=(0, 0, 0))
        self.ax.set_yscale(self.YScale)
        self.ax.set_ylim((self.YRange[0], self.YRange[1]))
        self.ax.legend()
        self.ax.set_axis_off()
        plt.gcf().canvas.draw_idle()
        plt.gcf().canvas.start_event_loop(self.DrawWait)

    def Value(self, label=None):
        if label == None:
            return sum(x[0][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][0][-1]

    def CumulativeValue(self, label=None):
        if label == None:
            return sum(x[1][-1] for x in self.Functions.values())
        else:
            return self.Functions[label][1][-1]

    def Print(self, digits=5):
        output = ""
        for name in self.Functions.keys():
            output = (
                output
                + name
                + ": "
                + str(round(self.CumulativeValue(name), digits))
                + " "
            )
        print(output)
