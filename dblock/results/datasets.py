from block.results.datasets import BaseDatasetResultsBuilder

import dblock.trainer


class DDatasetResultsBuilder(BaseDatasetResultsBuilder):

    def __init__(self, models_root, dataset, batch_size=256, build_activity=True):
        super().__init__(models_root, dataset, batch_size, build_activity, dblock.trainer.Trainer.hyperparams_mapper, dblock.trainer.Trainer.model_loader)
