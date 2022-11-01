import numpy as np

from dblock.models.builder import LinearDModel
from dblock.models.implementations.standard import BaseModel


class DBaseModel(BaseModel):

    def __init__(self, d, recurrent, method, t_len, n_neurons, n_layers=1, detach_recurrent_spikes=False, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, single_spike=True):
        super().__init__(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, recurrent, n_layers, n_neurons)
        self._d = d
        self._detach_recurrent_spikes = detach_recurrent_spikes

    @property
    def hyperparams(self):
        return {**super().hyperparams, "d": self._d, "detach_recurrent_spikes": self._detach_recurrent_spikes}


class NMNISTDModel(DBaseModel):

    def __init__(self, d, recurrent, method, t_len, n_neurons, n_layers=1, detach_recurrent_spikes=False, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, single_spike=True, **kwargs):
        print(f"d={d} recurrent={recurrent} method={method} t_len={t_len} n_layers={n_layers} detach_recurrent_spikes={detach_recurrent_spikes} heterogeneous_beta={heterogeneous_beta}")
        print(f"beta_requires_grad={beta_requires_grad} readout_max={readout_max} single_spike={single_spike}")
        print(f"kwargs={kwargs}")
        super().__init__(d, recurrent, method, t_len, n_neurons, n_layers, detach_recurrent_spikes, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        kwargs["detach_recurrent_spikes"] = detach_recurrent_spikes
        self._model = LinearDModel(d, recurrent, method, t_len, n_in=1156, n_out=10, n_hidden=n_neurons, n_layers=n_layers, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, scale=10, **kwargs)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class SHDDModel(DBaseModel):

    def __init__(self, d, recurrent, method, t_len, n_neurons, n_layers=1, detach_recurrent_spikes=False, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, single_spike=True, **kwargs):
        print(f"d={d} recurrent={recurrent} method={method} t_len={t_len} n_neurons={n_neurons} n_layers={n_layers} detach_recurrent_spikes={detach_recurrent_spikes} heterogeneous_beta={heterogeneous_beta}")
        print(f"beta_requires_grad={beta_requires_grad} readout_max={readout_max} single_spike={single_spike}")
        print(f"kwargs={kwargs}")
        super().__init__(d, recurrent, method, t_len, n_neurons, n_layers, detach_recurrent_spikes, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        kwargs["detach_recurrent_spikes"] = detach_recurrent_spikes
        self._model = LinearDModel(d, recurrent, method, t_len, n_in=700, n_out=20, n_hidden=n_neurons, n_layers=n_layers, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, scale=10, **kwargs)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)


class SSCDModel(DBaseModel):

    def __init__(self, d, recurrent, method, t_len, n_neurons, n_layers=1, detach_recurrent_spikes=False, heterogeneous_beta=False, beta_requires_grad=True, readout_max=True, single_spike=True, **kwargs):
        super().__init__(d, recurrent, method, t_len, n_neurons, n_layers, detach_recurrent_spikes, heterogeneous_beta, beta_requires_grad, readout_max, single_spike)
        kwargs["detach_recurrent_spikes"] = detach_recurrent_spikes
        self._model = LinearDModel(d, recurrent, method, t_len, n_in=700, n_out=35, n_hidden=n_neurons, n_layers=n_layers, hidden_beta=np.exp(-1/10), readout_beta=np.exp(-1/20), heterogeneous_beta=heterogeneous_beta, beta_requires_grad=beta_requires_grad, readout_max=readout_max, single_spike=single_spike, scale=10, **kwargs)

    def forward(self, spikes, return_all=False):
        return self._model(spikes, return_all)