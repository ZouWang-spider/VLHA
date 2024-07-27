import torch
import torch.nn as nn
import torch_geometric.nn as geo_nn


class GCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GCNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(geo_nn.GCNConv(input_size, hidden_size))  # 第一层使用 GCNConv

        for _ in range(1, num_layers):
            self.gcn_layers.append(geo_nn.GCNConv(hidden_size, hidden_size))  # 后续层继续使用 GCNConv

        # self.fc = nn.Linear(hidden_size, num_classes)  # 分类头

    def forward(self, node_features, edge_index):

        for layer in self.gcn_layers:
            node_features = layer(node_features, edge_index)

        # # 使用平均池化操作将单词级别的输出汇总为句子级别的输出
        # sentence_output = torch.mean(node_features, dim=0, keepdim=True)
        # logits = self.fc(sentence_output)

        return node_features



# #model use
# input_size = 2048   #768
# hidden_size = 32
# num_layers = 3
#
# # 创建模型实例
# model = GCNModel(input_size, hidden_size, num_layers)
#
# # 生成随机节点特征和边索引
# num_nodes = 10
# num_edges = 20
#
# #GCN 模型需要输入节点特征node_features与节点边索引edge_index
# node_features = torch.randn((num_nodes, input_size), dtype=torch.float32)
# edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
# print(node_features.shape)
# print(edge_index)
#
# # 测试模型
# output = model(node_features, edge_index)
# print(output.shape)