import torch
import torch.nn as nn
from brainbox.models import BBModel
import block.nn.methods as methods

from dblock.nn.layers import LinearNeurons, LinearDNeurons


class LinearModel(BBModel):

    def __init__(self, method, t_len, n_in, n_out, n_hidden, n_layers, skip_connections=False, hidden_beta=0.9, readout_beta=0.9, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, **kwargs):
        super().__init__()
        self._method = method
        self._t_len = t_len
        self._n_in = n_in
        self._n_out = n_out
        self._n_hidden = n_hidden
        self._n_layers = n_layers
        self._skip_connections = skip_connections
        self._hidden_beta = hidden_beta
        self._readout_beta = readout_beta
        self._heterogeneous_beta = heterogeneous_beta
        self._beta_requires_grad = beta_requires_grad
        self._readout_max = readout_max

        self._layers = nn.ModuleList()
        
        # Build hidden layers
        for i in range(n_layers):
            n_in = n_in if i == 0 else n_hidden
            self._layers.append(self._build_layer(n_in, n_hidden, hidden_beta, heterogeneous_beta, beta_requires_grad, **kwargs))

        self._readout_layer = self._build_layer(n_hidden, n_out, readout_beta, heterogeneous_beta, beta_requires_grad, single_spike=True, integrator=True)
        print(f"skip_connections={skip_connections}")
    @staticmethod
    def build_beta(beta_init, n, heterogeneous_beta):
        return [beta_init for _ in range(n if heterogeneous_beta else 1)]

    @property
    def hyperparams(self):
        return {**super().hyperparams, "method": self._method,  "t_len": self._t_len, "n_in": self._n_in,
                "n_out": self._n_out, "n_hidden": self._n_hidden, "n_layers": self._n_layers,
                "skip_connections": self._skip_connections, "hidden_beta": self._hidden_beta,
                "readout_beta": self._readout_beta, "heterogeneous_beta": self._heterogeneous_beta,
                "beta_requires_grad": self._beta_requires_grad, "readout_max": self._readout_max}

    def forward(self, spikes, return_all=False, deactivate_readout=False):

        spikes_list = []

        for i, layer in enumerate(self._layers):
            new_spikes = layer(spikes)

            if self._skip_connections and i > 0:
                #spikes = torch.clamp(new_spikes + spikes, min=0, max=1)
                spikes = new_spikes + spikes
            else:
                spikes = new_spikes

            if return_all:
                spikes_list.append(new_spikes)

        if deactivate_readout:
            return spikes

        spikes, mem = self._readout_layer(spikes, return_type=methods.RETURN_SPIKES_AND_MEM)
        output = torch.max(mem, 2)[0] if self._readout_max else torch.sum(mem, 2)

        if return_all:
            return output, spikes_list

        return output

    def _build_layer(self, n_in, n_out, beta_init, heterogeneous_beta, beta_requires_grad, **kwargs):
        print("building layer..")
        beta_init = LinearModel.build_beta(beta_init, n_out, heterogeneous_beta)
        return LinearNeurons(n_in, n_out, self._method, self._t_len, beta_init, beta_requires_grad, **kwargs)


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
