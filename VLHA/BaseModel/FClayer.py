import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_aspect_tags, num_sentiment_classes):
        super(FCLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_aspect_tags)
        self.fc4 = nn.Linear(num_aspect_tags, num_sentiment_classes)

    def forward(self, x, aspect_term_seq):
        # Reshape x to (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.shape

        x = x.view(-1, x.shape[-1])  # Reshape to (batch_size*seq_len, input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        print(x.shape)   #(10,3)

        # Convert aspect_term_seq to a tensor
        aspect_term_seq = torch.tensor(aspect_term_seq, dtype=torch.bool)

        # Extract corresponding values from x based on aspect_term_seq
        aspect_values = x[aspect_term_seq]

        # Perform average pooling
        if aspect_values.size(0) > 0:
            pooled_aspect_values = torch.mean(aspect_values, dim=0, keepdim=True)
        else:
            raise ValueError("No aspect terms found in the sequence.")

        # # Calculate sentiment polarity using softmax
        # sentiment_polarity = F.softmax(pooled_aspect_values, dim=1)
        #
        # # Determine the most probable sentiment class
        # sentiment_class = torch.argmax(sentiment_polarity, dim=1)


        return pooled_aspect_values


# # Example usage
# input_size = 768
# hidden_size1 = 512
# hidden_size2 = 256
# num_aspect_tags = 128
# num_sentiment_classes = 3
#
# model = FCLayer(input_size, hidden_size1, hidden_size2, num_aspect_tags, num_sentiment_classes)
# x = torch.randn(1, 10, 768)
# aspect = torch.randn(1, 128)
#
# output = model(x)   #预测目标tensor([2])
# print(output)
