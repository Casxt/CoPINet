import json
import os
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Dataset


class TaskSpecificARCDataset(Dataset):
    # 将字符映射为数字
    # pad_symbol 为填充标志
    # 1. closure-fill-color
    # 2. continue-link-point
    # 3. proximity-nearest-color
    # 4. region-fix-object
    # 5. similarity-same-shape
    # 6. symmetry-complete-rest
    # 7. symmetry-fixbroken
    WordMap = {k: v for v, k in
               enumerate(['pad_symbol', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'start_symbol'])}

    def __init__(self, index, dataset_path, method):
        self.index = index
        assert method == "train" or method == "test"
        self.method = method
        self.files = []
        self.dataset_path = dataset_path
        for dir_name, dirs, files in os.walk(Path(dataset_path)):
            files = filter(lambda f: ".json" in f, files)
            self.files.extend([Path(dir_name, file) for file in files])

    def __len__(self):
        _len = len(self.files)
        if self.method == 'train':
            return _len * 2
        else:
            return _len

    def __getitem__(self, index):
        """
        每个文件都包含n对训练用的input，target和m对测试用的input，target
        出于简单起见现在只取train或test的第一对
        """
        pad = TaskSpecificARCDataset.WordMap["pad_symbol"]
        if self.method == 'train':
            file_index, internal_index = index // 2, index % 2
            with open(self.files[file_index], 'r') as f:
                data = json.load(f)
            inputs = [i["input"] for i in data[self.method]]
            outputs = [i["output"] for i in data[self.method]]
            inputs = TaskSpecificARCDataset.pad_to(
                [torch.tensor(d) for d in inputs[3 * internal_index:3 * internal_index + 3]],
                [40, 40], pad_value=pad)
            outputs = TaskSpecificARCDataset.pad_to(
                [torch.tensor(d) for d in outputs[3 * internal_index:3 * internal_index + 3]],
                [40, 40], pad_value=pad)
        else:
            file_index, internal_index = index, 0
            with open(self.files[file_index], 'r') as f:
                data = json.load(f)
            inputs = [i["input"] for i in data[self.method]]
            outputs = [i["output"] for i in data[self.method]]
            inputs = TaskSpecificARCDataset.pad_to(
                [torch.tensor(d) for d in inputs[3 * internal_index:3 * internal_index + 3]],
                [40, 40], pad_value=pad)
            outputs = TaskSpecificARCDataset.pad_to(
                [torch.tensor(d) for d in outputs[3 * internal_index:3 * internal_index + 3]],
                [40, 40], pad_value=pad)

        input_data = torch.cat([
            inputs[0:1], torch.ones_like(inputs[0:1]) * pad, outputs[0:1],
            inputs[1:2], torch.ones_like(inputs[1:2]) * pad, outputs[1:2],
            inputs[2:3], torch.ones_like(inputs[2:3]) * pad,
            inputs[2:3], inputs[2:3], inputs[2:3], inputs[2:3], inputs[2:3], inputs[2:3], inputs[2:3], inputs[2:3],
        ], dim=0)
        target_data = outputs[2:3]
        task = torch.ones_like(input_data) * self.index
        task[input_data == pad] = pad
        return input_data, target_data, task

    @staticmethod
    def pad_to(batch, shape, pad_value=0):
        """将m个n维变量补齐至他们之中最大的那个"""
        max_shape = shape
        padded_tensor = []
        for item in batch:
            batch_dim = []
            for i, d in enumerate(item.shape):
                batch_dim.insert(0, max_shape[i] - d)  # 后面pad
                batch_dim.insert(0, 0)  # 前面不pad
            padded_tensor.append(
                torch.nn.functional.pad(item, batch_dim, mode='constant', value=pad_value).unsqueeze(0)
            )
        return torch.cat(padded_tensor, dim=0)

    @staticmethod
    def pad_nd(batch, pad_value=0):
        """将m个n维变量补齐至他们之中最大的那个"""
        max_shape = [0] * len(batch[0].shape)
        for item in batch:
            for i, d in enumerate(item.shape):
                if max_shape[i] < d:
                    max_shape[i] = d
        padded_tensor = []
        for item in batch:
            batch_dim = []
            for i, d in enumerate(item.shape):
                batch_dim.insert(0, max_shape[i] - d)  # 后面pad
                batch_dim.insert(0, 0)  # 前面不pad
            padded_tensor.append(
                torch.nn.functional.pad(item, batch_dim, mode='constant', value=pad_value).unsqueeze(0)
            )
        return torch.cat(padded_tensor, dim=0)

    @staticmethod
    def collate_fn(batch):
        """
        padding and generate mask
        data (b, len)， data[:, 0] 为开始标识符， data[:, -1] 为结束标识符
        mask (b, 1, len) 设置为1是方便广播以和attention(b, len, len)进行mask
        在multihead情况下, mask在attention模块中会进一步变换为(b, 1, 1, len)以和multihead-attention(b, head, len, len)进行mask

        targets[:, :-1] 为网络训练输入
        targets[:, 1:, :] 为网络训练预期输出
        """
        item_len = len(batch[0])
        raw_inputs, raw_targets, raw_answers = [[b[i] for b in batch] for i in range(item_len)]
        inputs = TaskSpecificARCDataset.pad_nd(raw_inputs, pad_value=TaskSpecificARCDataset.WordMap['pad_symbol'])
        inputs_mask = inputs.ne(TaskSpecificARCDataset.WordMap['pad_symbol']).to(torch.float)
        targets = TaskSpecificARCDataset.pad_nd(raw_targets, pad_value=TaskSpecificARCDataset.WordMap['pad_symbol'])
        targets_mask = targets.ne(TaskSpecificARCDataset.WordMap['pad_symbol']).to(torch.float)
        task = TaskSpecificARCDataset.pad_nd(raw_answers, pad_value=TaskSpecificARCDataset.WordMap['pad_symbol'])
        task_mask = task.ne(TaskSpecificARCDataset.WordMap['pad_symbol']).to(torch.float)
        return inputs, inputs_mask, targets, targets_mask, task, task_mask
