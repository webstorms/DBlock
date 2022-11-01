import numpy as np
from brainbox.models import BBModel

from block.models.builder import LinearModel


class BaseModel(BBModel):

    def __init__(self, method, t_len, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, single_spike=True, recurrent=False, n_layers=1, n_neurons=None):
        super().__init__()
        self._method = method
        self._t_len = t_len
        self._heterogeneous_beta = heterogeneous_beta
        self._beta_requires_grad = beta_requires_grad
        self._readout_max = readout_max
        self._single_spike = single_spike
        self._recurrent = recurrent
        self._n_layers = n_layers
        self._n_neurons = n_neurons

    @property
    def hyperparams(self):
        return {**super().hyperparams, "method": self._method, "t_len": self._t_len, "heterogeneous_beta": self._heterogeneous_beta, "beta_requires_grad": self._beta_requires_grad, "readout_max": self._readout_max, "single_spike": self._single_spike, "recurrent": self._recurrent, "n_layers": self._n_layers, "n_neurons": self._n_neurons}


class NMNISTModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True, recurrent=False, n_layers=1, n_neurons=128):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, recurrent, n_layers, n_neurons)
        self._model = LinearModel(method, t_len, n_in=1156, n_out=10, n_hidden=n_neurons, n_layers=n_layers, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, recurrent=recurrent)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class SHDModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True, recurrent=False, n_layers=1, n_neurons=128):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, recurrent, n_layers, n_neurons)
        self._model = LinearModel(method, t_len, n_in=700, n_out=20, n_hidden=n_neurons, n_layers=n_layers, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, recurrent=recurrent)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class SSCModel(BaseModel):

    def __init__(self, method, t_len, heterogeneous_beta=True, beta_requires_grad=True, readout_max=False, single_spike=True, recurrent=False, n_layers=1, n_neurons=128):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, recurrent, n_layers, n_neurons)
        self._model = LinearModel(method, t_len, n_in=700, n_out=35, n_hidden=n_neurons, n_layers=n_layers, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, recurrent=recurrent)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)