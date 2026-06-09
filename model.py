import torch
import torch.nn as nn


class Net(nn.Module):
    """面向 EEG 二分类的轻量卷积网络（EEGNet 风格）。

    相对最初版本的两处经过跨会话(LSO)实测验证的改进：
      1. 首层时间卷积核 64 -> 96（temporal_kernel），扩大时间感受野，跨会话均值 +1.1pt。
      2. 特征层用 SpatialDropout(nn.Dropout2d) 替代普通 Dropout，整张特征图一起丢弃，
         对卷积特征是更合适的正则，跨会话均值 +1.0pt。
    两者叠加把诚实跨会话准确率从 0.6535 提升到约 0.6718（详见实验报告）。
    增容类改动（更宽/更深/SE/多尺度）在 1694 样本规模下均无显著收益，故不采用。
    """

    def __init__(
        self,
        input_shape=(1, 59, 282),
        num_classes: int = 2,
        dropout: float = 0.35,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        temporal_kernel: int = 96,
    ):
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"expected input shape like (1, 59, 282), got {input_shape}")

        _, num_channels, num_timepoints = input_shape
        spatial_filters = temporal_filters * depth_multiplier
        temporal_pad = temporal_kernel // 2

        self.features = nn.Sequential(
            # 时间卷积：沿时间轴提取局部波形/节律模式
            nn.Conv2d(1, temporal_filters, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_pad), bias=False),
            nn.BatchNorm2d(temporal_filters),
            # 空间卷积：跨全部 59 通道学习通道组合关系（分组=深度可分离）
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
            nn.Dropout2d(dropout),  # SpatialDropout：按特征图整体丢弃
            # 深度可分离时序卷积：更高层时间特征
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
            nn.Dropout2d(dropout),  # SpatialDropout
        )

        with torch.no_grad():
            sample = torch.zeros(1, *input_shape)
            feature_dim = self.features(sample).flatten(start_dim=1).shape[1]

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
