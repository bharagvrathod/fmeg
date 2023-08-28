from torch import nn
import torch


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
                                     out_channels=10 * self.num_hessian_maps, kernel_size=(7, 7), padding=pad)
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
            hessian_map = self.hessian(feature_map)
            hessian_map = hessian_map.reshape(final_shape[0], self.num_hessian_maps, 10, final_shape[2],
                                              final_shape[3])
            heatmap = heatmap.unsqueeze(2)

            hessian = heatmap * hessian_map
            hessian = hessian.view(final_shape[0], final_shape[1], 10, -1)

            hessian_matrices = []
            for i in range(0, 10, 3):
                hessian_matrix = hessian[:, :, i:i+3, :]
                hessian_matrix = hessian_matrix.sum(dim=-1)
                hessian_matrices.append(hessian_matrix)

            hessian_matrices = torch.stack(hessian_matrices, dim=-1)
            hessian_matrices = hessian_matrices.view(hessian_matrices.shape[0], hessian_matrices.shape[1], 3, 3, -1)
            hessian_matrices = hessian_matrices.sum(dim=-1)
            
            out['hessian'] = hessian_matrices

        return out
