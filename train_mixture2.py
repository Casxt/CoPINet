import sys
from os import path

# sys.path.append(path.join(path.dirname(__file__), '..', ".."))
# sys.path.append("/root/abstract-reasoning-model")

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from dataset import ARCDataset, ContextARCDataset, MixtureDataset
# from model import ContextAttention
# from script.ContextAttention.tools import *
from src import ContextARCDataset, MixtureDataset, SimpleCoPINet, to_device, compute_mask_accuracy, \
    compute_element_accuracy, compute_corrects_accuracy, compute_task_accuracy, compute_balance_loss, compute_task_loss, \
    loging

device = 0
epochs = 200
batchSize = 32
workernum = 4
# torch.autograd.set_detect_anomaly(True)
subPath = Path("copinet/mixture/6th_train")
save = Path("/home/zhangkai/abstract-reasoning/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/home/zhangkai/abstract-reasoning/log", subPath))

# 1. closure-fill-color
# 2. continue-link-point
# 3. proximity-nearest-color
# 4. region-fix-object
# 5. similarity-same-shape
# 6. symmetry-complete-rest
# 7. symmetry-fixbroken

dataset_paths = [
    (1, Path("/home/zhangkai/abstract-reasoning/training-large/closure-fill-color/")),
    (2, Path("/home/zhangkai/abstract-reasoning/training-large/continue-link-point/")),
    (3, Path("/home/zhangkai/abstract-reasoning/training-large/proximity-nearest-color/")),
    (4, Path("/home/zhangkai/abstract-reasoning/training-large/region-fix-object/")),
    (5, Path("/home/zhangkai/abstract-reasoning/training-large/similarity-same-shape/")),
    (6, Path("/home/zhangkai/abstract-reasoning/training-large/symmetry-complete-rest/")),
    (7, Path("/home/zhangkai/abstract-reasoning/training-large/symmetry-fixbroken/"))
]
ecnnet = SimpleCoPINet(num_attr=len(ContextARCDataset.WordMap), num_rule=7, channel_out=len(ContextARCDataset.WordMap))

# ecnnet.load_state_dict(torch.load(
#     "/root/abstract-reasoning-model/weight/pretrain-encoder-attention/mixture/1st_pretrain/epoch58-acc0.981920063495636.weight",
#     map_location=torch.device('cpu')), strict=False)

net = torch.nn.DataParallel(ecnnet, device_ids=[0, 1, 2, 3]).cuda(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), )

# 0.65^25 = 0.00002
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9, last_epoch=-1)

padding = ContextARCDataset.WordMap['pad_symbol']


def train_forward(net, inputs, inputs_mask, ctx_x, ctx_x_mask, ctx_y, ctx_y_mask):
    b, h, w = inputs.shape
    inputs = torch.cat([d.squeeze(1).to(torch.float) for d in (ctx_x, ctx_y, inputs, inputs)], dim=1)
    output = net(inputs)
    task = torch.zeros((b, 8), device=inputs.device, dtype=torch.float)
    return output, task


def val_forward(net, inputs, inputs_mask, ctx_x, ctx_x_mask, ctx_y, ctx_y_mask):
    b, h, w = inputs.shape
    inputs = torch.cat([d.squeeze(1).to(torch.float) for d in (ctx_x, ctx_y, inputs, inputs)], dim=1)
    output = net(inputs)
    _, res = torch.max(output, dim=-1)
    task = torch.zeros((b), device=inputs.device)
    return res, task


def to_same_size(pad_value, *batch):
    max_shape = [batch[0].shape[0], 40, 40] * len(batch[0].shape)
    # for item in batch:
    #     for i, d in enumerate(item.shape):
    #         if max_shape[i] < d:
    #             max_shape[i] = d
    padded_tensor = []
    for item in batch:
        batch_dim = []
        for i, d in enumerate(item.shape):
            batch_dim.insert(0, max_shape[i] - d)  # 后面pad
            batch_dim.insert(0, 0)  # 前面不pad
        padded_tensor.append(
            torch.nn.functional.pad(item, batch_dim, mode='constant', value=pad_value)
        )
    return padded_tensor


step, val_step = 0, 0

for epoch in range(epochs):
    net.train()
    datasets = []
    for i, dataset_path in dataset_paths:
        datasets.append(ContextARCDataset(index=i, dataset_path=Path(dataset_path), method='train'))
    dataset = MixtureDataset(datasets)
    train = DataLoader(dataset, shuffle=True, num_workers=workernum, batch_size=batchSize,
                       collate_fn=ContextARCDataset.collate_fn)
    for index, batch in enumerate(train):
        if index > 3500:  # 提前退出快速进行val
            break
        inputs, ctx_input, targets, ctx_targets, task = to_same_size(padding, *to_device(device, *batch))

        start_time = time.time()
        step += len(inputs)

        answers = targets.ne(inputs).to(torch.long)
        answers[inputs.eq(padding)] = padding

        outputs, predict_task = train_forward(net, inputs, inputs.ne(padding).to(torch.float),
                                              ctx_input, ctx_input.ne(padding).to(torch.float),
                                              ctx_targets, ctx_targets.ne(padding).to(torch.float))

        loss = compute_balance_loss(outputs, targets, answers, padding) + compute_task_loss(predict_task, task[:, 0, 0],
                                                                                            padding)

        _, predict_task = torch.max(predict_task, dim=-1)
        _, output_index = torch.max(outputs, dim=-1)
        output_index = output_index.to(outputs.dtype)
        task_accuracy = compute_task_accuracy(predict_task, task[:, 0, 0], padding)
        element_accuracy = compute_element_accuracy(output_index, targets, padding)
        mask_accuracy = compute_mask_accuracy(inputs, output_index, targets, padding)
        correct_accuracy = compute_corrects_accuracy(output_index, targets, padding)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loging(writer, 'train', epoch, step, time.time() - start_time, dataset_size=len(dataset),
               batch_size=len(inputs),
               **{'loss': loss, 'element_accuracy': element_accuracy, 'correct_accuracy': correct_accuracy,
                  'mask_accuracy': mask_accuracy, 'task_accuracy': task_accuracy})
    # 更新学习率
    scheduler.step()

    with torch.no_grad():
        net.eval()
        for i, dataset_path in dataset_paths:
            dataser_name = dataset_path.parts[-1]
            dataset = ContextARCDataset(index=i, dataset_path=Path(dataset_path),
                                        method='test')  # ARCDataset(Path(dataset_path), method='test')
            val = DataLoader(dataset, shuffle=False, pin_memory=False, num_workers=workernum,
                             batch_size=batchSize, collate_fn=ContextARCDataset.collate_fn)

            start_time = used_time = time.time()
            total_element_accuracy = torch.tensor(0.)
            total_correct_accuracy = torch.tensor(0.)
            total_mask_accuracy = torch.tensor(0.)
            total_task_accuracy = torch.tensor(0.)
            index = 0
            for index, batch in enumerate(val):
                if index > 100:
                    break
                inputs, ctx_input, targets, ctx_targets, task = to_same_size(padding, *to_device(device, *batch))

                start_time = time.time()
                val_step += len(inputs)

                output_index, predict_task = val_forward(net, inputs, inputs.ne(padding).to(torch.float),
                                                         ctx_input, ctx_input.ne(padding).to(torch.float),
                                                         ctx_targets, ctx_targets.ne(padding).to(torch.float))

                # answers = targets.ne(inputs).to(torch.long)
                # answers[inputs.eq(padding)] = padding
                #
                task_accuracy = compute_task_accuracy(predict_task, task[:, 0, 0], padding)
                element_accuracy = compute_element_accuracy(output_index, targets, padding)
                mask_accuracy = compute_mask_accuracy(inputs, output_index, targets, padding)
                correct_accuracy = compute_corrects_accuracy(output_index, targets, padding)
                total_element_accuracy = total_element_accuracy + element_accuracy
                total_correct_accuracy = total_correct_accuracy + correct_accuracy
                total_mask_accuracy = total_mask_accuracy + mask_accuracy
                total_task_accuracy = total_task_accuracy + task_accuracy
                loging(None, f'{dataser_name}-val-step', epoch, val_step, time.time() - start_time,
                       dataset_size=len(val.dataset),
                       batch_size=len(inputs),
                       **{'element_accuracy': element_accuracy, 'correct_accuracy': correct_accuracy,
                          'mask_accuracy': mask_accuracy, 'task_accuracy': task_accuracy})
            index += 1
            total_element_accuracy = total_element_accuracy / index
            total_correct_accuracy = total_correct_accuracy / index
            total_mask_accuracy = total_mask_accuracy / index
            total_task_accuracy = total_task_accuracy / index
            loging(writer, f'{dataser_name}-val', epoch, epoch, time.time() - start_time,
                   dataset_size=len(train.dataset),
                   batch_size=len(val.dataset),
                   **{'element_accuracy': total_element_accuracy, 'correct_accuracy': total_correct_accuracy,
                      'mask_accuracy': total_mask_accuracy, 'task_accuracy': total_task_accuracy})

        torch.save(
            net.module.state_dict(),
            Path(save,
                 f"epoch{epoch}-acc{total_element_accuracy}.weight")
        )
