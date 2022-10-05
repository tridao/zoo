import pytest

import torch

from src.datamodules.associative_recall import AssociativeRecallDataset, AssociativeRecallDataModule


class TestAssociativeRecall:

    def test_output(self):
        batch_size = 64
        max_length = 1000
        seed = 2357
        dataset = AssociativeRecallDataset(100, seqlen=13, seed=seed)
        print(dataset[0])
        print(dataset[1])
        print(dataset[2])

        datamodule = AssociativeRecallDataModule(seqlen=13, seed=seed)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        x, y = next(iter(train_loader))
        print(x, y)
