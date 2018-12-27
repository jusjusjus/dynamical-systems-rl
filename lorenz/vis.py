
import numpy as np
from gym.envs.classic_control import rendering


class Trace:

    def __init__(self, offset=0.0, marker=None):
        self._trace = rendering.make_polyline([[0,0]])
        self.marker = marker
        if self.marker is not None:
            self._hline = rendering.Line(start=(0,0), end=(0,0))
        else:
            self._hline = None
        self.offset = offset

    def update(self, data):
        MIN = data.min()
        rng = (data.max()-MIN)+10**-6
        self._trace.v = zip(range(data.size), (data-MIN)/rng+self.offset)
        if self._hline:
            marker = (self.marker-MIN)/rng+self.offset
            self._hline.start = (0, marker)
            self._hline.end = (data.size, marker)

    def __call__(self, renderer):
        renderer.add_geom(self._trace)
        if self._hline:
            renderer.add_geom(self._hline)


class Renderer(rendering.Viewer):

    _x_start = 0

    def __init__(self, max_steps, screen_width=800, screen_height=400):
        # from gym.envs.classic_control import rendering
        super().__init__(screen_width, screen_height)
        self.max_steps = max_steps

    def get_traces(self, num_traces):
        if hasattr(self, 'traces'):
            assert len(self.traces) == num_traces
        else:
            self._time = np.arange(self.max_steps)
            self.set_bounds(self._x_start, self.max_steps, 0, 7)
            self.traces = [
                Trace(offset=(num_traces-1-i)*1.1, marker=0.0)
                for i in range(num_traces)
            ]
            for t, trace in enumerate(self.traces):
                trace(self)
        return self.traces

    def update_data(self, signals):
        traces = self.get_traces(signals.shape[0])
        for trace, data in zip(traces, signals):
            trace.update(data)

    def render(self, mode='human'):
        return super().render(return_rgb_array = mode=='rgb_array')
