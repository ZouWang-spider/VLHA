import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

        # 线性映射层定义
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        # 最终映射层
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, score_matrix):
        # 执行线性映射以获取 Q, K, V
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        # 根据得分矩阵调整注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) + score_matrix

        # 计算 softmax 归一化的注意力权重
        attention_weights = F.softmax(attention_scores / torch.sqrt(torch.tensor(self.d_head, dtype=torch.float32)),
                                      dim=-1)

        # 加权平均得到输出
        attention_output = torch.matmul(attention_weights, V)

        # 最终线性映射
        attention_output = self.final_linear(attention_output)

        return attention_output


class Cross_TransformerModel(nn.Module):
    def __init__(self, num_heads, d_model, dff, dropout_rate=0.1):
        super(Cross_TransformerModel, self).__init__()

        self.attention = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.dense1 = nn.Linear(d_model, dff)
        self.dense2 = nn.Linear(dff, d_model)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text_tensor, visual_tensor, score_matrix, text_tensor2, visual_tensor2, score_matrix2):
        # Structural alignment section 文本作为 Query，视觉作为 Key 和 Value
        query = text_tensor  # 没有添加额外维度，因为在 MultiHeadAttention 中会处理
        key_value = visual_tensor.unsqueeze(0)  # 添加 batch_size 维度

        # 调用 MultiHeadAttention 计算注意力
        attention_scores = self.attention(query, key_value, key_value, score_matrix)
        attention_output = self.dropout(attention_scores)
        # 残差连接和层归一化
        out1 = self.layernorm1(query + attention_output)

        #Semantic Alignment Section 文本作为 Query，视觉作为 Key 和 Value
        query = text_tensor2  # 没有添加额外维度，因为在 MultiHeadAttention 中会处理
        key_value = visual_tensor2.unsqueeze(0)  # 添加 batch_size 维度

        # 调用 MultiHeadAttention 计算注意力
        attention_scores = self.attention(query, key_value, key_value, score_matrix2)
        attention_output = self.dropout(attention_scores)
        # 残差连接和层归一化
        out2 = self.layernorm1(query + attention_output)

        # 在第二维度上拼接这两个张量
        concat = torch.cat((out1, out2), dim=2)
        # 使用池化层将维度从 [1, 10, 768*2] 池化为 [1, 10, 768]
        pooling_layer = nn.MaxPool1d(kernel_size=2, stride=2)  # 以最大池化为例
        concat = pooling_layer(concat.permute(0, 1, 2)).permute(0, 1, 2)

        # 全连接层和最终层归一化
        output = self.dense2(F.relu(self.dense1(concat)))
        output = self.dropout(output)
        output = self.layernorm2(concat + output)

        return output


# # 创建 Transformer 模型实例
# transformer_model = TransformerModel(num_heads=8, d_model=768, dff=2048)
#
# # 假设这是你的数据
# text_tensor = torch.randn(10, 768)  # 文本特征张量
# visual_tensor = torch.randn(4, 768)  # 视觉特征张量
# score_matrix = torch.randn(10, 4)  # 得分矩阵
#
# text_tensor2 = torch.randn(10, 768)  # 文本特征张量
# visual_tensor2 = torch.randn(4, 768)  # 视觉特征张量
# score_matrix2 = torch.randn(10, 4)  # 得分矩阵
#
# # 输入数据到 Transformer 模型
# output = transformer_model(text_tensor, visual_tensor, score_matrix, text_tensor2, visual_tensor2, score_matrix2)
#
# # 输出的处理可以根据具体任务进一步调整
# print("Output shape:", output.shape)
