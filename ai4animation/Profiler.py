import cProfile
import io
import pstats
import time


class Profiler:
    """
    Usage:
    1. As a context manager (like cProfile):
       with profiler:
           # code to profile

    2. Manual start/end:
       profiler.Start()
       # code to profile
       profiler.End()

    3. Automatic periodic printing:
       profiler.Check()
       # profiler will print stats every print_interval seconds
    """

    def __init__(self, print_interval=1.0):
        self.instance = None
        self.enabled = False
        self.context_mode = False

        self.print_interval = print_interval
        self.last_print_time = 0.0

        self.stats_buffer = io.StringIO()

    def __enter__(self):
        """Context manager entry"""
        self.context_mode = True
        self.instance = cProfile.Profile()
        self.instance.enable()
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.enabled and self.instance is not None:
            self.instance.disable()
            self.enabled = False
            self.instance = None
            self.context_mode = False

    def Start(self):
        if not self.enabled:
            self.instance = cProfile.Profile()
            self.instance.enable()
            self.enabled = True
            self.last_print_time = time.time()

    def End(self):
        if self.enabled and self.instance is not None:
            self.instance.disable()
            self.enabled = False
            self.instance = None

    def Check(self, top_n=100):
        if not self.enabled:
            self.Start()

        if not self.enabled or self.instance is None or self.context_mode:
            return

        current_time = time.time()
        if current_time - self.last_print_time >= self.print_interval:
            self._print_stats(top_n)
            self.last_print_time = current_time

    def _print_stats(self, top_n):
        if self.instance is None:
            return

        self.stats_buffer.seek(0)
        self.stats_buffer.truncate(0)

        stats = pstats.Stats(self.instance, stream=self.stats_buffer)
        stats.sort_stats("cumulative")
        stats.print_stats(top_n)  # Print top n functions by cumulative time

        # Get stats
        stats_content = self.stats_buffer.getvalue()

        if stats_content.strip():
            print("\n" + "=" * 50)
            print("__UPDATE__ PROFILER STATS")
            print("=" * 50)
            print(stats_content)
            print("=" * 50 + "\n")

    def IsEnabled(self):
        return self.enabled
