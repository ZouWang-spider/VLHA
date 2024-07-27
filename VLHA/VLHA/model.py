import nltk
import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import Linear, CrossEntropyLoss
from sklearn.metrics import log_loss

from VLHA.DataProcess.DatasetPross import load_dataset
from VLHA.BaseModel.GCNmodel import GCNModel
from VLHA.Module.StructuralAlignment import  Visual_graph_compute, Syntax_Graph_compute, AlignmentModel
from VLHA.BaseModel.SyntacticGraph import BERT_Embedding
from VLHA.Module.SemanticAlignment import visual_seq_feature, Semantic_AlignmentModel
from VLHA.BaseModel.CrossTransformer import Cross_TransformerModel
from VLHA.BaseModel.Crfmodel import CRFModel, CRFModel2
from VLHA.BaseModel.FClayer import FCLayer


#加载数据
data_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015/train.txt"
image_path = "E:/PythonProject2/VisionLanguageMABSA/Datasets/twitter2015_images"
samples = load_dataset(data_path,image_path)
def get_dataset(sample):
    image_path = sample['image_path']
    sentence = sample['sentence']
    aspect_terms = sample['aspect_term']
    sentiments = sample['sentiment']
    return image_path, sentence, aspect_terms, sentiments

#将数据转化为0-1二进制
def convert_aspect_terms_to_binary_sequence(sentence, aspect_terms):
    # 使用nltk库进行分词
    tokens = nltk.word_tokenize(sentence)
    # 创建一个与tokens相同长度的二进制序列，初始值为0
    binary_sequence = [0] * len(tokens)
    # 将方面词的位置标记为1
    start_idx = sentence.find(aspect_terms)
    end_idx = start_idx + len(aspect_terms)
    aspect_tokens = nltk.word_tokenize(sentence[:start_idx]) + nltk.word_tokenize(aspect_terms)
    start_token_idx = len(nltk.word_tokenize(' '.join(aspect_tokens[:-1])))
    end_token_idx = start_token_idx + len(nltk.word_tokenize(aspect_terms))
    # 将方面词对应的位置标记为1
    binary_sequence[start_token_idx-1:end_token_idx-1] = [1] * (end_token_idx - start_token_idx)

    return binary_sequence


#model initial
#Visual GCN model
input_size = 2048   #768
hidden_size = 768
num_layers = 2
visual_model = GCNModel(input_size, hidden_size, num_layers)

#Text GCN model
input_size = 768   #768
hidden_size = 768
num_layers = 2
text_model = GCNModel(input_size, hidden_size, num_layers)


Struc_model = AlignmentModel()
semantic_model = Semantic_AlignmentModel()

# 创建 Transformer 模型实例
transformer_model = Cross_TransformerModel(num_heads=8, d_model=768, dff=2048)

#CRF layer
crfmodel = CRFModel(2)
crfmodel2 = CRFModel2(2)


#FC layer
input_size = 768
hidden_size1 = 512
hidden_size2 = 256
num_aspect_tags = 128
num_sentiment_classes = 3
fcmodel = FCLayer(input_size, hidden_size1, hidden_size2, num_aspect_tags, num_sentiment_classes)



if __name__ == "__main__":
    num_epochs = 50
    batch_size = 32
    criterion = CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_aspect_accuracy = 0.0
        total_sentiment__accuracy = 0.0
        total_accuracy = 0.0
        total_aspect_loss = 0.0
        total_sentiment_loss = 0.0
        total_loss2 = 0.0
        # Iterate through batches
        for start_idx in range(0, len(samples), batch_size):
            end_idx = start_idx + batch_size
            batch_samples = samples[start_idx:end_idx]

            # datasets Train
            for sample in batch_samples:
                try:
                    image_path, sentence, aspect_terms, sentiments = get_dataset(sample)  # get dataset,image and text

                    # 转化为序列形式
                    aspect_term_seq = convert_aspect_terms_to_binary_sequence(sentence, aspect_terms)

                    # Structural alignment section
                    visual_output = Visual_graph_compute(image_path, 0.7, visual_model)  # (n, 768)
                    text_output = Syntax_Graph_compute(sentence, text_model)  # (l,768))
                    score_matrix1 = Struc_model(text_output, visual_output)  # score matrix1 (l,n)
                    # print(score_matrix1.shape)

                    # Semantic Alignment Section
                    image_feature = visual_seq_feature(image_path)  # (n,768)
                    text_feature = BERT_Embedding(sentence)  # (l,768)
                    score_matrix2 = semantic_model(text_feature, image_feature)  # score matrax2 (l,n)
                    # print(score_matrix2.shape)

                    # 输入数据到 Transformer 模型
                    output = transformer_model(text_output, visual_output, score_matrix1, text_feature, image_feature,
                                               score_matrix2)

                    NNmodel = nn.Sequential(
                        nn.Linear(768, 2),
                        nn.Softmax(dim=1)
                    )

                    output_tensor = NNmodel(output)  # (10,2)

                except Exception as e:
                  print(f"Exception occurred: {str(e)}")
                  continue  # 发生异常时跳出当前循环，执行下一个循

                # 真实序列标签转化为张量tensor([[0,0,0,1,1,0,0,0]])
                aspect_term_seq_tensor = torch.tensor([aspect_term_seq])

                # 计算准确率 accuracy
                # 使用 CRF 层解码，获取最优路径（预测的标签序列）
                tags_pred = crfmodel2.decode(output)
                # 计算准确率 Accuracy
                tags_pred = torch.tensor(tags_pred).squeeze(0)  # 将预测的标签序列转换为张量并去除批次维度
                correct_predictions = (tags_pred == aspect_term_seq_tensor).sum().item()  # 计算正确预测的数量
                total_predictions = aspect_term_seq_tensor.numel()  # 计算总的标签数量
                aspect_accuracy = correct_predictions / total_predictions  # 计算准确率
                total_aspect_accuracy += aspect_accuracy

                # 计算损失 loss
                try:
                  aspect_loss = crfmodel(output_tensor, aspect_term_seq_tensor)
                except Exception as e:
                  print(f"Exception occurred: {str(e)}")
                  continue  # 发生异常时跳出当前循环，执行下一个循
                total_aspect_loss += aspect_loss

                # 全连接层  输出方面词对应的情感极性
                sentiment_class = fcmodel(output, aspect_term_seq)

                # 映射标签到类别索引
                sentiment = [sentiments]
                labels = torch.tensor(sentiment, dtype=torch.long)  # 情感标签
                label_mapping = {-1: 0, 0: 1, 1: 2}
                mapped_labels = torch.tensor([label_mapping[label.item()] for label in labels], dtype=torch.long)
                # 计算损失（例如，使用交叉熵损失）loss
                sentiment_loss = criterion(sentiment_class, mapped_labels)
                # print(sentiment_loss)
                total_sentiment_loss += sentiment_loss
                total_loss = aspect_loss + sentiment_loss
                # 总损失
                total_loss2 += aspect_loss + sentiment_loss

                # 计算预测准确率 Accuracy
                predicted_labels = torch.argmax(sentiment_class, dim=1)
                correct_predictions = (predicted_labels == mapped_labels).sum().item()
                total_predictions = mapped_labels.size(0)
                sentiment_accuracy = correct_predictions / total_predictions

                total_sentiment__accuracy += sentiment_accuracy
                # 总准确率
                total_accuracy += aspect_accuracy + sentiment_accuracy

                # Set up the optimizer
                all_parameters = list(visual_model.parameters()) + list(text_model.parameters()) + list(Struc_model.parameters()) +  list(semantic_model.parameters()) +\
                                 list(transformer_model.parameters()) + list(fcmodel.parameters()) +  list(crfmodel.parameters()) + list(crfmodel2.parameters())
                optimizer = Adam(all_parameters, lr=0.00002)
                scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                # Backward pass and optimization step
                optimizer.zero_grad()
                aspect_loss.backward(retain_graph=True)
                sentiment_loss.backward()
                optimizer.step()

                print(f'Epoch {epoch + 1}/{num_epochs}, aspec_accuracy: {aspect_accuracy:.4f}, sentiment_accuracy:{sentiment_accuracy:.4f}')
                print(f'Epoch {epoch + 1}/{num_epochs}, aspect_loss:{aspect_loss}, sentiment_loss:{sentiment_loss}')
        print(f'total_accuracy: {total_accuracy/len(samples)}, total_aspect_accuracy: {total_aspect_accuracy/len(samples)},total_sentiment__accuracy: {total_sentiment__accuracy/len(samples)}')
        print(f'total loss: {total_loss2/len(samples)}, total_aspect_loss: {total_aspect_loss/len(samples)}, total_sentiment_loss: {total_sentiment_loss/len(samples)}')







