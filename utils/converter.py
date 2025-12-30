"""CTC Label Converter for text recognition"""

import torch


class CTCLabelConverter:
    """CTC 标签转换器"""
    
    def __init__(self, character_set='0123456789abcdefghijklmnopqrstuvwxyz'):
        self.character = '-' + character_set  # blank + characters
        self.dict = {char: idx for idx, char in enumerate(self.character)}
    
    def encode(self, text_list):
        batch_size = len(text_list)
        
        encoded = []
        for text in text_list:
            text = text.lower()
            label = [self.dict. get(char, 0) for char in text]
            encoded.append(label)
        
        lengths = [len(label) for label in encoded]
        max_len = max(lengths)
        
        labels = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, label in enumerate(encoded):
            labels[i, : len(label)] = torch.LongTensor(label)
        
        return labels, torch.LongTensor(lengths)
    
    def decode(self, preds, raw=False):
        if preds.dim() == 3:
            preds = preds. argmax(dim=2)
        
        preds = preds.cpu().numpy()
        batch_size = preds.shape[0]
        
        text_list = []
        for i in range(batch_size):
            pred = preds[i]
            
            if raw:
                chars = [self.character[idx] for idx in pred]
                text_list.append(''.join(chars))
            else:
                char_list = []
                prev_char = None
                for idx in pred:
                    if idx == 0:
                        prev_char = None
                        continue
                    char = self.character[idx]
                    if char != prev_char:
                        char_list.append(char)
                    prev_char = char
                text_list.append(''.join(char_list))
        
        return text_list
