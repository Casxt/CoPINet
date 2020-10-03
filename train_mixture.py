import sys
from os import path

# sys.path.append(path.join(path.dirname(__file__), '..', ".."))
# sys.path.append("/root/abstract-reasoning-model")

import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from dataset import ARCDataset, TaskSpecificARCDataset, MixtureDataset
from src import TaskSpecificARCDataset, MixtureDataset, CoPINet, to_device, compute_mask_accuracy, \
    compute_element_accuracy, compute_corrects_accuracy, compute_balance_loss, loging
# from script.PretrainAbstarctAttention.tools import *

device = 0
epochs = 200
batchSize = 8

workernum = 4
# torch.autograd.set_detect_anomaly(True)
subPath = Path("copinet/mixture/1st_train")
save = Path("/root/abstract-reasoning-model/weight", subPath)
save.mkdir(parents=True) if not save.exists() else None

writer = SummaryWriter(Path("/root/abstract-reasoning-model/log", subPath))

# 1. closure-fill-color
# 2. continue-link-point
# 3. proximity-nearest-color
# 4. region-fix-object
# 5. similarity-same-shape
# 6. symmetry-complete-rest
# 7. symmetry-fixbroken

dataset_paths = [
    (1, Path("/root/abstract-reasoning-model/pretrain_dataset/closure-fill-color/")),
    (2, Path("/root/abstract-reasoning-model/pretrain_dataset/continue-link-point/")),
    (4, Path("/root/abstract-reasoning-model/pretrain_dataset/region-fix-object/")),
    (5, Path("/root/abstract-reasoning-model/pretrain_dataset/similarity-same-shape/")),
    (6, Path("/root/abstract-reasoning-model/pretrain_dataset/symmetry-complete-rest/"))
]
ecnnet = CoPINet(num_attr=len(TaskSpecificARCDataset.WordMap), num_rule=5, channel_out=len(TaskSpecificARCDataset.WordMap))

ecnnet.load_state_dict(torch.load(
    "/root/abstract-reasoning-model/weight/pretrain-encoder-attention/mixture/1st_pretrain/epoch58-acc0.981920063495636.weight",
    map_location=torch.device('cpu')), strict=False)

net = torch.nn.DataParallel(ecnnet, device_ids=[0, 1, 2, 3]).cuda(device)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), )

# 0.65^25 = 0.00002
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.85, last_epoch=-1)


def train_forward(net: CoPINet, inputs, inputs_mask, task, task_mask):
    inputs = torch.nn.functional.interpolate(inputs.view(batchSize, 16, 40, 40), scale_factor=2, mode='nearest')
    output = net(inputs)
    return output


def val_forward(net: CoPINet, inputs, inputs_mask, task, task_mask):
    inputs = torch.nn.functional.interpolate(inputs.view(batchSize, 16, 40, 40), scale_factor=2, mode='nearest')
    output = net(inputs)
    _, res = torch.max(output, dim=-1)
    return res


def random_mask(data: torch.Tensor, mask_symbol, pad_symbol):
    # 10% 几率被mask
    m = torch.rand(*data.shape, device=data.device).le(0.1)
    p = data.eq(pad_symbol)
    res = data.clone()
    # 设置随机mask
    res[m] = mask_symbol
    # 修复pad
    res[p] = pad_symbol
    return res


step, val_step = 0, 0
for epoch in range(epochs):
    net.train()
    datasets = []
    for i, dataset_path in dataset_paths:
        datasets.append(TaskSpecificARCDataset(index=i, dataset_path=Path(dataset_path), method='train'))
    dataset = MixtureDataset(datasets)
    train = DataLoader(dataset, shuffle=True, num_workers=workernum, batch_size=batchSize,
                       collate_fn=TaskSpecificARCDataset.collate_fn)
    for index, batch in enumerate(train):
        if index > 3500:  # 提前退出快速进行val
            break
        inputs, inputs_mask, targets, targets_mask, task, task_mask = to_device(device, *batch)
        # answers = inputs
        start_time = time.time()
        step += len(inputs)
        inp = inputs[:, 6:7]
        # output is (b, d1, d2, ..., dn, c) c is onehot logits

        # inputs = random_mask(targets, ARCDataset.WordMap["start_symbol"], ARCDataset.WordMap['pad_symbol'])

        answers = targets.ne(inp).to(torch.long)
        print(inputs.shape, inputs_mask.shape)
        answers[inputs_mask[:, 6:7].eq(False)] = TaskSpecificARCDataset.WordMap['pad_symbol']

        outputs = train_forward(net, inp, inputs_mask, task, task_mask)
        loss = compute_balance_loss(outputs, targets, answers, TaskSpecificARCDataset.WordMap['pad_symbol'])
        _, output_index = torch.max(outputs, dim=-1)
        output_index = output_index.to(outputs.dtype)

        element_accuracy = compute_element_accuracy(output_index, targets, TaskSpecificARCDataset.WordMap['pad_symbol'])
        mask_accuracy = compute_mask_accuracy(inp, output_index, targets, TaskSpecificARCDataset.WordMap['pad_symbol'])
        correct_accuracy = compute_corrects_accuracy(output_index, targets, TaskSpecificARCDataset.WordMap['pad_symbol'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loging(writer, 'train', epoch, step, time.time() - start_time, dataset_size=len(dataset),
               batch_size=len(inputs),
               **{'loss': loss, 'element_accuracy': element_accuracy, 'correct_accuracy': correct_accuracy,
                  'mask_accuracy': mask_accuracy})
    # 更新学习率
    scheduler.step()

    with torch.no_grad():
        net.eval()
        for i, dataset_path in dataset_paths:
            dataser_name = dataset_path.parts[-1]
            dataset = TaskSpecificARCDataset(index=i, dataset_path=Path(dataset_path),
                                             method='test')  # ARCDataset(Path(dataset_path), method='test')
            val = DataLoader(dataset, shuffle=False, pin_memory=False, num_workers=workernum,
                             batch_size=batchSize, collate_fn=TaskSpecificARCDataset.collate_fn)

            start_time = used_time = time.time()
            total_element_accuracy = torch.tensor(0.)
            total_correct_accuracy = torch.tensor(0.)
            total_mask_accuracy = torch.tensor(0.)
            index = 0
            for index, batch in enumerate(val):
                if index > 100:
                    break
                inputs, inputs_mask, targets, targets_mask, task, task_mask = to_device(device, *batch)
                # answers = inputs
                start_time = time.time()
                val_step += len(inputs)

                # inputs = random_mask(targets, ARCDataset.WordMap["start_symbol"], ARCDataset.WordMap['pad_symbol'])
                # answers = targets.ne(inputs).to(torch.long)
                # answers[inputs_mask.eq(False)] = ARCDataset.WordMap['pad_symbol']

                output_index = val_forward(net, inputs, inputs_mask, task, task_mask)
                inp = inputs[:, 6:7]
                element_accuracy = compute_element_accuracy(output_index, targets, TaskSpecificARCDataset.WordMap['pad_symbol'])
                mask_accuracy = compute_mask_accuracy(inp, output_index, targets, TaskSpecificARCDataset.WordMap['pad_symbol'])
                correct_accuracy = compute_corrects_accuracy(output_index, targets, TaskSpecificARCDataset.WordMap['pad_symbol'])
                total_element_accuracy = total_element_accuracy + element_accuracy
                total_correct_accuracy = total_correct_accuracy + correct_accuracy
                total_mask_accuracy = total_mask_accuracy + mask_accuracy
                loging(None, f'{dataser_name}-val-step', epoch, val_step, time.time() - start_time,
                       dataset_size=len(val.dataset),
                       batch_size=len(inputs),
                       **{'element_accuracy': element_accuracy, 'correct_accuracy': correct_accuracy,
                          'mask_accuracy': mask_accuracy})
            index += 1
            total_element_accuracy = total_element_accuracy / index
            total_correct_accuracy = total_correct_accuracy / index
            total_mask_accuracy = total_mask_accuracy / index
            loging(writer, f'{dataser_name}-val', epoch, epoch, time.time() - start_time,
                   dataset_size=len(train.dataset),
                   batch_size=len(val.dataset),
                   **{'element_accuracy': total_element_accuracy, 'correct_accuracy': total_correct_accuracy,
                      'mask_accuracy': total_mask_accuracy})

        torch.save(
            net.module.state_dict(),
            Path(save,
                 f"epoch{epoch}-acc{total_element_accuracy}.weight")
        )
