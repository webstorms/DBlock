from block.benchmark import LayerBenchmarker
from dblock.models import builder


class dLayerBenchmarker(LayerBenchmarker):

    def __init__(self, d, recurrent, method, t_len, n_in, n_hidden, n_layers, heterogeneous_beta, beta_requires_grad=False, min_r=0, max_r=200, n_samples=11, batch_size=16):
        super().__init__(method, t_len, n_in, n_hidden, n_layers, heterogeneous_beta, beta_requires_grad, min_r, max_r, n_samples, batch_size)
        self._d = d
        self._recurrent = recurrent
        self._model = builder.LinearDModel(d, recurrent, method, t_len, n_in, 1, n_hidden, n_layers, heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, single_spike=True)

    def _get_description(self):
        return {**super()._get_description(), "d": self._d, "recurrent": self._recurrent}

    def _get_df_name(self):
        return f"{super()._get_df_name()}_{self._d}_{self._recurrent}"
