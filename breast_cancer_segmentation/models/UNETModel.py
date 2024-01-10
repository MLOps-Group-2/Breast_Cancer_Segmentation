import torch
import monai

class UNETModel(monai.networks.nets.UNet):
    def __init__(self,
                 loss_function,
                 learning_rate=1e-3,
                 spatial_dims=0,
                 in_channels=0,
                 out_channels=0,
                 channels=(),
                 strides=(),
                 num_res_units=2
                 ):
        # create UNet, DiceLoss and Adam optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units
        ).to(self.device)

        self.loss_function = loss_function
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


