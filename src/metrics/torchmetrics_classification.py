'''Wrap torchmetrics classification metrics to cast the targets to longs.'''
from torchmetrics import F1Score, AUROC, Precision

class MyF1Score(F1Score):
    def update(self, preds, target):
        super().update(preds, target.long())

class MyAUROC(AUROC):
    def update(self, preds, target):
        super().update(preds, target.long())

class MyPrecision(Precision):
    def update(self, preds, target):
        super().update(preds, target.long())