from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from ._functional import soft_dice_score, to_tensor
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["ConcetricDiceLoss"]


class ConcetricDiceLoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        square_weights: Optional[List[int]] = None,
        input_shape : Optional[tuple] = None,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(ConcetricDiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.square_weights = square_weights
        H,W = input_shape
        self.square_mask = self.create_mask((H,W))
        self.intermediate_score = None

    def create_mask(self, shape, multiplier=2):

        num_inst = len(self.square_weights)

        square_indexes = range(num_inst)
        H, W = shape[0], shape[1]
        mask = torch.zeros((H, W))
        mask[0:H,0:W] = square_indexes[0]
        x_0,y_0,x_1,y_1 = 0,0,W,H

        values = 12
        for event in range(num_inst-1):
            y_0 += H//(2*multiplier)
            x_0 += W//(2*multiplier)

            y_1 -= H//(2*multiplier)
            x_1 -= H//(2*multiplier)

            mask[y_0:y_1,x_0:x_1] = square_indexes[event+1]

            H, W = int(y_1-y_0), int(x_1 - x_0)
            values += 2

        return mask
    
    def forward(self, y_pred_org: torch.Tensor, y_true_org: torch.Tensor) -> torch.Tensor:

        assert y_true_org.size(0) == y_pred_org.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred_org = y_pred_org.log_softmax(dim=1).exp()
            else:
                y_pred_org = F.logsigmoid(y_pred_org).exp()

        bs = y_true_org.size(0)
        num_classes = y_pred_org.size(1)
        dims = (0, 2)

        ## Mask factor normalization
        intermediate_scores = []
        scores = 0

        import numpy as np
        np.save("/mnt/prj001/Bibhabasu_Mohapatra/github/experiments/dump.npy",self.square_mask.numpy(),)
        for idx in range(len(self.square_weights)):
            H,W = torch.where(self.square_mask == idx)

            y_pred = y_pred_org[...,H,W]*self.square_weights[idx]
            y_true = y_true_org[...,H,W]

            if self.mode == BINARY_MODE:
                
                y_true = y_true.view(bs, 1, -1)
                y_pred = y_pred.view(bs, 1, -1)

                if self.ignore_index is not None:
                    mask = y_true != self.ignore_index
                    y_pred = y_pred * mask
                    y_true = y_true * mask

            if self.mode == MULTICLASS_MODE:
                y_true = y_true.view(bs, -1)
                y_pred = y_pred.view(bs, num_classes, -1)

                if self.ignore_index is not None:
                    mask = y_true != self.ignore_index
                    y_pred = y_pred * mask.unsqueeze(1)

                    y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                    y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
                else:
                    y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                    y_true = y_true.permute(0, 2, 1)  # N, C, H*W

            if self.mode == MULTILABEL_MODE:
                y_true = y_true.view(bs, num_classes, -1)
                y_pred = y_pred.view(bs, num_classes, -1)

                if self.ignore_index is not None:
                    mask = y_true != self.ignore_index
                    y_pred = y_pred * mask
                    y_true = y_true * mask


            scores += self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
            intermediate_scores.append(scores)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]

        self.intermediate_score = intermediate_scores
        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)
    
