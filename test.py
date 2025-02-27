import torch
from torch import nn
from torchvision import models
from torchsummary import summary

from models.backbone import BackboneBase_VGG
from models.p2pnet import RegressionModel, ClassificationModel, AnchorPoints, Decoder
from util.misc import NestedTensor


# 这是一个 P2PNet 类的假设定义，你需要替换为你自己的模型定义
class P2PNet(nn.Module):
    def __init__(self, backbone, row=2, line=2):
        super().__init__()
        self.backbone = backbone
        self.num_classes = 2
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(num_features_in=256, num_anchor_points=num_anchor_points)
        self.classification = ClassificationModel(num_features_in=256, \
                                                  num_classes=self.num_classes, \
                                                  num_anchor_points=num_anchor_points)

        self.anchor_points = AnchorPoints(pyramid_levels=[3, ], row=row, line=line)

        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])

        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[1]) * 100  # 8x
        classification = self.classification(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction
        output_coord = regression + anchor_points
        output_class = classification
        out = {'pred_logits': output_class, 'pred_points': output_coord}

        return out





# 1. 加载一个预训练的 ResNet 作为 backbone
backbone = models.vgg16(pretrained=True)

# 2. 初始化 P2PNet 模型，并将 ResNet 作为 backbone 传递给模型
model_p2pnet = P2PNet(backbone=backbone)

# 3. 将模型设置为评估模式
model_p2pnet.eval()

# 4. 使用 torchsummary 查看参数量，假设输入图像大小为 (3, 224, 224)
summary(model_p2pnet, input_size=(3, 224, 224))
