from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and Hessian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_hessian=False, scale_factor=1,
                 single_hessian_map=False, pad=0):
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_hessian:
            self.num_hessian_maps = 1 if single_hessian_map else num_kp
            self.hessian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                     out_channels=4 * self.num_hessian_maps, kernel_size=(7, 7), padding=pad)
            self.hessian.weight.data.zero_()
            self.hessian.bias.data.zero_()
        else:
            self.hessian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}

        return kp

    def forward(self, x):
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.hessian is not None:
            # Assuming self.hessian contains the second-order derivatives
            hessian = self.hessian(feature_map)
            # Adjust the shape using reshape instead of view
            hessian = hessian.reshape(final_shape[0], -1, 2, 2)  # Adjust the shape as needed

            out['hessian'] = hessian

        return out









