import torch
import torch.nn as nn
from torchcrf import CRF

# # 假设你的数据和标签
# features = torch.randn(1, 10, 2)  # 假设有一个样本，序列长度为10，每个特征是768维的向量
# tags_true = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 0, 1, 1]])  # 真实的标签序列

# 定义 CRF 模型
class CRFModel(nn.Module):
    def __init__(self, num_tags):
        super(CRFModel, self).__init__()
        self.crf = CRF(num_tags)  # 初始化 CRF 层，指定标签的数量

    def forward(self, features, tags_true):
        # 计算 CRF 损失
        loss = self.crf(features, tags_true)

        return -loss  # 返回负对数似然损失

class CRFModel2(nn.Module):
    def __init__(self, num_tags):
        super(CRFModel2, self).__init__()
        self.fc = nn.Linear(768, num_tags)  # 将特征维度转换为标签维度
        self.crf = CRF(num_tags)  # 初始化 CRF 层，指定标签的数量

    def forward(self, features):
        emissions = self.fc(features)  # 计算发射分数
        return emissions

    def compute_loss(self, features, tags_true):
        emissions = self.forward(features)
        loss = -self.crf(emissions, tags_true)  # CRF 的负对数似然损失
        return loss

    def decode(self, features):
        emissions = self.forward(features)
        return self.crf.decode(emissions)  # 使用 CRF 层进行解码，获取最优路径