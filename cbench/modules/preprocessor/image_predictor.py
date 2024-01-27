import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from .base import Preprocessor
from cbench.nn.base import NNTrainableModule
from cbench.nn.layers import MaskedConv2d, MaskedConv3d

from cbench.ar import ar_linear_op
from cbench.ar import autoregressive_transform_3way_op_tpl

# TODO: a general preditor
class ImagePredictorPreprocessor(Preprocessor, NNTrainableModule):
    def __init__(self, predictor_offsets : List[Tuple[int]], *args, **kwargs):
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

        self.predictor_offsets = predictor_offsets
        self.predictor_model = nn.Linear(len(predictor_offsets), 1)

    def _preprocess_data(self, data):
        return data

    def _postprocess_data(self, data):
        return data

    # def _stack_prediction_data(self, data, offset):

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def preprocess(self, data, *args, prior=None, **kwargs):
        return super().preprocess(data, *args, prior=prior, **kwargs)

    def postprocess(self, data, *args, prior=None, **kwargs):
        # TODO: c++ impl may be faster
        return data


# TODO: reparent to ImagePredictorPreprocessor
class ThreeWayAutoregressivePreprocessor(Preprocessor, NNTrainableModule):
    def __init__(self, *args,
        add_dist_loss=True,
        dist_loss_weight=1.0,
        dist_loss_threshold=0.0,
        detach_pred_for_dist_loss=True,
        fixed_weights=None,
        fixed_biases=None,
        **kwargs):
        self.predictor_offsets = [
            [(0, -1, -1),
            (0, 0, -1),
            (0, -1, 0),],
            [(-1, 0, 0),
            (-1, 0, -1),
            (0, 0, -1),],
            [(-1, 0, 0),
            (-1, 0, -1),
            (0, 0, -1),],
        ]
        super().__init__(*args, **kwargs)
        NNTrainableModule.__init__(self)

        self.add_dist_loss = add_dist_loss
        self.dist_loss_weight = dist_loss_weight
        self.dist_loss_threshold = dist_loss_threshold
        self.detach_pred_for_dist_loss = detach_pred_for_dist_loss

        predictor_models = []
        for pred in self.predictor_offsets:
            predictor_models.append(nn.Linear(len(pred), 1))
        self.predictor_models = nn.ModuleList(predictor_models)

        if fixed_weights is not None or fixed_biases is not None:
            assert fixed_weights is not None and fixed_biases is not None
            for model, weight in zip(self.predictor_models, fixed_weights):
                model.weight.data = torch.as_tensor(weight).reshape(1, 3)
                model.weight.requires_grad = False
            for model, bias in zip(self.predictor_models, fixed_biases):
                model.bias.data = torch.as_tensor(bias)
                model.bias.requires_grad = False

    def _get_prediction(self, data):
        pred = torch.zeros_like(data)
        data = F.pad(data, (1, 0, 1, 0), "constant", 0)
        for c in range(3):
            if c==0:
                pred_src = torch.stack([
                    data[..., c:(c+1), :-1, :-1],
                    data[..., c:(c+1), 1:, :-1],
                    data[..., c:(c+1), :-1, 1:],
                ], dim=-1)
            else:
                pred_src = torch.stack([
                    data[..., (c-1):c, 1:, 1:],
                    data[..., (c-1):c, 1:, :-1],
                    data[..., c:(c+1), 1:, :-1],
                ], dim=-1)
            pred[..., c, :, :] = self.predictor_models[c](pred_src.reshape(-1, 3)).reshape_as(pred[..., c, :, :])
        return pred

    def forward(self, data, *args, prior=None, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        pred = self._get_prediction(data)
        
        pred_dist = F.mse_loss(data, pred) #, reduction='none')
        if self.add_dist_loss and self.training:
            if pred_dist.mean() < self.dist_loss_threshold:
                pred_dist = torch.zeros_like(pred_dist)
            if self.detach_pred_for_dist_loss:
                pred = pred.detach() # no need to backward diff
            self.update_cache("loss_dict", predictor_loss=pred_dist.mean() * self.dist_loss_weight)
        self.update_cache("metric_dict", predictor_distance=pred_dist.mean())

        # pred_clamp = pred.clamp(min=0, max=1)
        diff = data - pred + 0.5
        # if self.add_dist_loss:
        #     diff = diff.detach() # no need to backward diff
        # similar to uint8 rounding: [-0.5, 0] to [0.5, 0], [1.0, 1.5] to [1.0, 0.5]
        # diff = diff + 0.5
        # diff[diff < 0] = -diff[diff < 0]
        # diff[diff > 1] = 2 - diff[diff > 1]
        # try cosine based uint8 simulation?

        if not self.training:
            # data_quant = (data*255).round().byte().type_as(data).div(255)
            # pred_quant = (pred*255).round().byte().type_as(data).div(255)
            # data_uint8 = (data_quant*255).round().byte()
            # pred_uint8 = (pred*255).round().byte()
            # diff_uint8 = (data_quant*255).round() - (pred*255).round().long() + 128
            # diff = diff_uint8.type_as(pred).div(255)
            # diff = (data*255).round().div(255) - (pred*255).round().div(255) + 0.5
            # diff = (diff*255).round().byte().type_as(data).div(255)
            data_uint8 = (data*255).round().byte()
            # diff_uint8 = data_uint8 - pred_uint8 + 128
            diff_uint8 = data_uint8 - (pred*255).round().long() + 128
            # self._diff_cache = diff_uint8
            diff = diff_uint8.type_as(pred).div(255)

        # TODO: a soft clamp that enable gradient passthrough
        return diff # .clamp(min=0, max=1)

    def preprocess(self, data, *args, prior=None, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        # quantize data
        data_quant = (data*255).round().byte().type_as(data).div(255)

        assert(data.shape[-3] == 3)
        pred = self._get_prediction(data_quant)
        # pred[..., 0, 0, :] = 0.0
        # pred[..., 0, :, 0] = 0.0
        # self._data_cache = data_quant
        # self._pred_cache = pred

        # TODO: define min and max as parameters
        # pred_uint8 = (pred*255).round().byte()
        data_uint8 = (data_quant*255).round().byte()
        # diff_uint8 = data_uint8 - pred_uint8 + 128
        diff_uint8 = data_uint8 - (pred*255).round().long() + 128
        # self._diff_cache = diff_uint8
        diff = diff_uint8.type_as(pred).div(255)

        # pred_clamp = pred.clamp(min=0, max=1)
        # diff = data - pred_clamp + 0.5
        # # similar to uint8 rounding: [-0.5, 0] to [0.5, 0], [1.0, 1.5] to [1.0, 0.5]
        # diff[diff < 0] = -diff[diff < 0]
        # diff[diff > 1] = 2 - diff[diff > 1]
        # return diff.clamp(min=0, max=1)
        
        # diff = data_quant - (pred*255).round().div(255) + 0.5
        return diff


    def postprocess(self, data, *args, prior=None, **kwargs):
        # handle device
        if data.device != self.device:
            data = data.to(device=self.device)

        assert(data.shape[-3] == 3)
        diff = data # - 0.5
        diff_uint8 = (diff*255).round().byte()
        # TODO: c++ impl may be faster (check correctness!)
        weights = [self.predictor_models[i].weight for i in range(3)]
        biases = [self.predictor_models[i].bias for i in range(3)]
        ar_funcs = [ar_linear_op(w.reshape(-1).tolist(),float(b.item())) for w,b in zip(weights, biases)]
        recovered_data = autoregressive_transform_3way_op_tpl(data.float().detach().cpu().numpy(), ar_funcs[0], self.predictor_offsets[0])
        recovered_data = torch.as_tensor(recovered_data)

        # recovered_data = torch.zeros_like(data)
        # # recovered_pred = torch.zeros_like(data)
        # recovered_data = F.pad(recovered_data, (1, 0, 1, 0), "constant", 0)
        # for c in range(data.shape[-3]):
        #     if c==0:
        #         for h in range(1, data.shape[-2]+1):
        #             for w in range(1, data.shape[-1]+1):
        #                 pred_src = torch.stack([
        #                     recovered_data[..., c:(c+1), h-1, w-1],
        #                     recovered_data[..., c:(c+1), h, w-1],
        #                     recovered_data[..., c:(c+1), h-1, w],
        #                 ], dim=-1)
        #                 pred = self.predictor_models[c](pred_src.reshape(-1, 3)).reshape(*diff.shape[:-3])
        #                 pred_uint8 = (pred*255).round().byte()
        #                 data_uint8 = diff_uint8[..., c, h-1, w-1] - 128 + pred_uint8
        #                 # recovered_pred[..., c, h-1, w-1] = pred.reshape(*diff.shape[:-3])
        #                 recovered_data[..., c, h, w] = data_uint8.type_as(recovered_data).div(255)
        #     else:
        #         for w in range(1, data.shape[-1]+1):
        #             pred_src = torch.stack([
        #                 recovered_data[..., (c-1):c, 1:, w],
        #                 recovered_data[..., (c-1):c, 1:, w-1],
        #                 recovered_data[..., c:(c+1), 1:, w-1],
        #             ], dim=-1)
        #             pred = self.predictor_models[c](pred_src.reshape(-1, 3)).reshape(*diff.shape[:-3], data.shape[-2])
        #             pred_uint8 = (pred*255).round().byte()
        #             data_uint8 = diff_uint8[..., c, :, w-1] - 128 + pred_uint8
        #             # recovered_pred[..., c, :, w-1] = pred.reshape(*diff.shape[:-3], data.shape[-2])
        #             recovered_data[..., c, 1:, w] = data_uint8.type_as(recovered_data).div(255)
        # recovered_data = recovered_data[..., :, 1:, 1:]
        # assert (self._data_cache == recovered_data).all()
        return recovered_data.type_as(data)
