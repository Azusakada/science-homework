import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(
        self,
        input_shape=(1, 59, 282),
        num_classes: int = 2,
        dropout: float = 0.35,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"expected input shape like (1, 59, 282), got {input_shape}")

        _, num_channels, num_timepoints = input_shape
        spatial_filters = temporal_filters * depth_multiplier

        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                kernel_size=(num_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            feature_dim = self.features(sample).flatten(start_dim=1).shape[1]

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
