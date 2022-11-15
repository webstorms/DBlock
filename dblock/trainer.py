import logging

from block import trainer as block_trainer
from brainbox import trainer

from dblock import models


class Trainer(block_trainer.Trainer):

    def __init__(self, root, model, dataset, n_epochs, batch_size, lr, milestones, gamma=0.1, val_dataset=None, device="cuda", track_activity=False):
        super().__init__(root, model, dataset, n_epochs, batch_size, lr, milestones, gamma, val_dataset, device, track_activity)

    def on_epoch_complete(self, save):
        if save:
            self.save_model_log()

            epoch_loss = self.log["train_loss"][-1]
            if epoch_loss < self._min_loss:
                logging.info(f"Saving model...")
                self._min_loss = epoch_loss
                self.save_model()

        n_epoch = len(self.log["train_loss"])

        if n_epoch == self._milestones[self._milestone_idx]:
            logging.info(f"Decaying lr...")
            self.lr *= self._gamma
            # Load best model
            self.model = Trainer.load_model(self.root, self.id, self.device, self.dtype)
            self.optimizer = self.optimizer_func(
                self.model.parameters(), self.lr, **self.optimizer_kwargs
            )

            if self._milestone_idx != len(self._milestones) - 1:
                logging.info(f"New milestone target...")
                self._milestone_idx += 1

    @staticmethod
    def load_model(root, id, device, dtype):
        return trainer.load_model(root, id, Trainer.model_loader, device, dtype)

    @staticmethod
    def model_loader(hyperparams):
        model_params = hyperparams["model"]

        name = model_params["name"]
        method = model_params["method"]
        t_len = int(model_params["t_len"])
        heterogeneous_beta = bool(model_params["heterogeneous_beta"])
        beta_requires_grad = bool(model_params["beta_requires_grad"])
        readout_max = bool(model_params["readout_max"])
        single_spike = bool(model_params.get("single_spike", True))
        d = model_params.get("d")
        recurrent = model_params.get("recurrent")
        n_layers = model_params.get("n_layers")
        n_neurons = model_params.get("n_neurons")
        detach_recurrent_spikes = bool(model_params.get("detach_recurrent_spikes"))
        skip_connections = bool(model_params.get("skip_connections"))

        if name == "NMNISTModel":
            return models.NMNISTModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, recurrent, n_layers, n_neurons)
        elif name == "SHDModel":
            return models.SHDModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, recurrent, n_layers, n_neurons)
        elif name == "SSCModel":
            return models.SSCModel(method, t_len, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, recurrent, n_layers, n_neurons)

        elif name == "NMNISTDModel":
            return models.NMNISTDModel(d, recurrent, method, t_len, n_neurons, n_layers, detach_recurrent_spikes, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, skip_connections=skip_connections)
        elif name == "SHDDModel":
            return models.SHDDModel(d, recurrent, method, t_len, n_neurons, n_layers, detach_recurrent_spikes, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, skip_connections=skip_connections)
        elif name == "SSCDModel":
            return models.SSCDModel(d, recurrent, method, t_len, n_neurons, n_layers, detach_recurrent_spikes, heterogeneous_beta, beta_requires_grad, readout_max, single_spike, skip_connections=skip_connections)

        print("bas bas bas bas bas bad")

    @staticmethod
    def hyperparams_mapper(hyperparams):
        method = hyperparams["model"]["method"]
        heterogeneous_beta = hyperparams["model"]["heterogeneous_beta"]
        beta_requires_grad = hyperparams["model"]["beta_requires_grad"]
        readout_max = hyperparams["model"]["readout_max"]
        d = hyperparams["model"].get("d", None)
        n_layers = hyperparams["model"].get("n_layers", None)
        recurrent = hyperparams["model"]["recurrent"]
        detach_recurrent_spikes = hyperparams["model"].get("detach_recurrent_spikes", None)

        return {"method": method, "heterogeneous_beta": heterogeneous_beta, "beta_requires_grad": beta_requires_grad, "readout_max": readout_max, "d": d, "n_layers": n_layers, "recurrent": recurrent, "detach_recurrent_spikes": detach_recurrent_spikes}
