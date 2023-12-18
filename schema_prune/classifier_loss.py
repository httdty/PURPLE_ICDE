"""
MIT License

Copyright (c) 2023 RUC KBReasoning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# CrossEntropyLoss = softmax + log + NLLLoss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.5, reduction=None):
        super(FocalLoss, self).__init__()

        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        assert input_tensor.shape[0] == target_tensor.shape[0]

        prob = F.softmax(input_tensor, dim = -1)
        log_prob = torch.log(prob + 1e-8)

        loss = F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )

        return loss

class ClassifierLoss():
    def __init__(self, alpha, gamma):
        weight = torch.FloatTensor([1-alpha, alpha])
        if torch.cuda.is_available():
            weight = weight.cuda()

        self.focal_loss = FocalLoss(
            weight = weight,
            gamma = gamma,
            reduction = 'mean'
        )

        # self.ce_loss = nn.CrossEntropyLoss(weight = weight, reduction = "mean")

    def compute_batch_loss(self, batch_logits, batch_labels, batch_size):
        loss = 0
        for logits, labels in zip(batch_logits, batch_labels):
            loss += self.focal_loss(logits, labels)

        return loss/batch_size

    def compute_loss(
        self,
        batch_table_name_cls_logits,
        batch_table_labels,
        batch_column_info_cls_logits,
        batch_column_labels
    ):
        batch_size = len(batch_table_labels)

        table_loss = self.compute_batch_loss(batch_table_name_cls_logits, batch_table_labels, batch_size)
        column_loss = self.compute_batch_loss(batch_column_info_cls_logits, batch_column_labels, batch_size)

        return table_loss + column_loss