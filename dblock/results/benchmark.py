from block.results.benchmark import BaseBenchmarkQuery


class dBenchmarkdvstQuery(BaseBenchmarkQuery):
    
    def __init__(self, root, batches=[32, 64, 128]):
        super().__init__(root, batches)

    def get_speedups(self, apply_mean_time=False, **kwargs):
        results_df = self._query_results(**kwargs)

        vanilla_times = results_df[results_df["method"] == "standard"].set_index(["t_len", "batch"])[["forward_time", "backward_time", "total_time"]]
        fast_times = results_df[results_df["method"] == "fast_naive"].set_index(["t_len", "d", "batch"])[["forward_time", "backward_time", "total_time"]]

        if apply_mean_time:
            vanilla_times = vanilla_times.groupby(["t_len", "batch"]).mean()
            fast_times = fast_times.groupby(["t_len", "d", "batch"]).mean()

        speedup_df = vanilla_times / fast_times
        speedup_df.rename(columns={"forward_time": "forward_speedup", "backward_time": "backward_speedup", "total_time": "total_speedup"}, inplace=True)

        return speedup_df


class dBenchmarkdvsnQuery(BaseBenchmarkQuery):

    def __init__(self, root, batches=[32, 64, 128]):
        super().__init__(root, batches)

    def get_speedups(self, apply_mean_time=False, **kwargs):
        results_df = self._query_results(**kwargs)

        vanilla_times = results_df[results_df["method"] == "standard"].set_index(["units", "batch"])[["forward_time", "backward_time", "total_time"]]
        fast_times = results_df[results_df["method"] == "fast_naive"].set_index(["units", "d", "batch"])[["forward_time", "backward_time", "total_time"]]

        if apply_mean_time:
            vanilla_times = vanilla_times.groupby(["units", "batch"]).mean()
            fast_times = fast_times.groupby(["units", "d", "batch"]).mean()

        speedup_df = vanilla_times / fast_times
        speedup_df.rename(columns={"forward_time": "forward_speedup", "backward_time": "backward_speedup", "total_time": "total_speedup"}, inplace=True)

        return speedup_df