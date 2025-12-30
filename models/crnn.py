"""
CRNN: Convolutional Recurrent Neural Network for Text Recognition

Reference: https://arxiv.org/abs/1507.05717
"""

import torch
import torch.nn as nn
import torch. nn.functional as F


class BidirectionalLSTM(nn.Module):
    """双向 LSTM 层"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.fc(recurrent)
        return output


class CRNN(nn.Module):
    """CRNN 文本识别模型"""
    
    def __init__(
        self,
        img_height=32,
        num_classes=37,
        hidden_size=256,
        use_mobilenet=False
    ):
        super().__init__()
        
        self. img_height = img_height
        
        if use_mobilenet:
            self.cnn = self._build_mobilenet_backbone()
            cnn_output_channels = 256
        else: 
            self.cnn = self._build_resnet_backbone()
            cnn_output_channels = 512
        
        self.map2seq = nn.Linear(cnn_output_channels, hidden_size)
        self.rnn1 = BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        self.rnn2 = BidirectionalLSTM(hidden_size, hidden_size, num_classes)
    
    def _build_resnet_backbone(self):
        return nn.Sequential(
            nn. Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn. Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def _build_mobilenet_backbone(self):
        def conv_bn_relu(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
                nn. BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        def depthwise_separable(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(inplace=True),
                nn. Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        
        return nn.Sequential(
            conv_bn_relu(3, 32, 2),
            depthwise_separable(32, 64, 1),
            depthwise_separable(64, 128, 2),
            depthwise_separable(128, 128, 1),
            depthwise_separable(128, 256, 2),
            depthwise_separable(256, 256, 1),
            nn.AdaptiveAvgPool2d((1, None))
        )
    
    def forward(self, x):
        conv = self.cnn(x)
        
        b, c, h, w = conv.size()
        if h > 1:
            conv = conv.view(b, c * h, w)
        else:
            conv = conv.squeeze(2)
        
        conv = conv.permute(0, 2, 1)
        seq = self.map2seq(conv)
        seq = self.rnn1(seq)
        output = self.rnn2(seq)
        
        return output
