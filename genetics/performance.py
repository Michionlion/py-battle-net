import time
import sys
import statistics as stats


class Benchmark:
    """Benchmark a given function using this class's static run method."""

    @staticmethod
    def run(function, output=False):
        """Benchmark function, optionally hiding or displaying output."""
        timings = []
        if not output:
            stdout = sys.stdout
        for i in range(100):
            if not output:
                sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            timings.append(seconds)
            mean = stats.mean(timings)
            if not output:
                sys.stdout = stdout
            if i < 10 or i % 10 == 9:
                print("Run: {0}  Mean: {1:3.2f}  Stdev: {2:3.2f}".format(
                    1 + i, mean,
                    stats.stdev(timings, mean) if i > 1 else 0))


#
