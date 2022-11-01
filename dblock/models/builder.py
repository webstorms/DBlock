from block.models.builder import LinearModel

from dblock.nn.layers import LinearDNeurons


class LinearDModel(LinearModel):

    def __init__(self, d, recurrent, method, t_len, n_in, n_out, n_hidden, n_layers, skip_connections=False, hidden_beta=0.9, readout_beta=0.9, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, **kwargs):
        self._d = d
        self._recurrent = recurrent
        super().__init__(method, t_len, n_in, n_out, n_hidden, n_layers, skip_connections, hidden_beta, readout_beta, heterogeneous_beta, beta_requires_grad, readout_max, **kwargs)
        self._readout_layer = self._build_layer(n_hidden, n_out, readout_beta, heterogeneous_beta, beta_requires_grad=True, recurrent=False, single_spike=True, integrator=True)

    @property
    def hyperparams(self):
        return {**super().hyperparams, "d": self._d, "recurrent": self._recurrent}

    def _build_layer(self, n_in, n_out, beta_init, heterogeneous_beta, beta_requires_grad, recurrent=None, **kwargs):
        print("building d2 layer..", kwargs)
        beta_init = LinearModel.build_beta(beta_init, n_out, heterogeneous_beta)
        recurrent = self._recurrent if recurrent is None else recurrent
        detach_recurrent_spikes = kwargs.get("detach_recurrent_spikes", True)
        use_recurrent_max = kwargs.get("use_recurrent_max", True)
        kwargs.pop("detach_recurrent_spikes", None)
        kwargs.pop("use_recurrent_max", None)

        return LinearDNeurons(n_in, n_out, self._d, recurrent, self._method, self._t_len, detach_recurrent_spikes, use_recurrent_max, beta_init, beta_requires_grad, **kwargs)
