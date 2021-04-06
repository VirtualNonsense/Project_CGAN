import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, number_of_gpus: int, feature_map_size: int, color_channels: int):
        super(Discriminator, self).__init__()
        self.number_of_gpus = number_of_gpus
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(color_channels, feature_map_size, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(feature_map_size * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0), bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_vector):
        return self.main(input_vector)
