import torch


def to_device(device, *tensors: torch.Tensor):
    return [t.cuda(device) for t in tensors]


def compute_balance_loss(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, padding_value):
    # target is (b, d1, d2, ..., dn), output is (b, d1, d2, ..., dn, c)
    # output to (b, c, l) shape
    # token_num = target.ne(padding_value).sum()
    target = target.view(output.shape[0], -1)
    mask = mask.view(output.shape[0], -1)
    output = output.view(output.shape[0], -1, output.shape[-1]).transpose(1, 2)

    positive_mask = mask.eq(1)
    positive_num = positive_mask.sum(dim=1, dtype=torch.float)
    negative_mask = mask.eq(0)
    negative_num = negative_mask.sum(dim=1, dtype=torch.float)
    total_num = positive_num + negative_num
    positive_weight = negative_num / total_num
    negative_weight = positive_num / total_num
    loss = torch.nn.NLLLoss(reduction='none', ignore_index=padding_value)(output, target)
    positive_loss = (loss * positive_mask).sum(dim=1) * positive_weight
    negative_loss = (loss * negative_mask).sum(dim=1) * negative_weight
    return (positive_loss + negative_loss).sum() / total_num.sum()


def compute_element_accuracy(output: torch.Tensor, target: torch.Tensor, padding_value):
    """
    逐batch统计相等元素数
    减去padding数目计算准确率
    最后取平均
    """
    # target is (b, d1, d2, ..., dn), output is (b, d1, d2, ..., dn)
    target = target.view(output.shape[0], -1)
    output = output.view(output.shape[0], -1)
    masks = target.eq(padding_value)
    paddings = masks.sum(dim=1, dtype=output.dtype)
    lens = masks.eq(0).sum(dim=1, dtype=output.dtype)

    # 修正padding元素
    output = output.clone()
    output[masks] = padding_value

    corrects = output.eq(target).sum(dim=1, dtype=torch.float)
    corrects = (corrects - paddings) / lens

    return torch.true_divide(corrects.sum(), corrects.nelement())


def compute_mask_accuracy(input: torch.Tensor, output: torch.Tensor, target: torch.Tensor, padding_value):
    """
    取output与target预测为1的并集
    取output与target预测为1的交集
    交集 / 并集
    """
    # input, target is (b, d1, d2, ..., dn), output is (b, d1, d2, ..., dn)
    target = target.view(output.shape[0], -1)
    input = input.view(output.shape[0], -1)
    output = output.view(output.shape[0], -1)
    padding_mask = target.eq(padding_value)
    output[padding_mask] = padding_value

    target_diff = target.ne(input)
    output_diff = output.ne(input)
    union_diff = target_diff.__or__(output_diff)
    output_correct = output.eq(target)
    masked_output_correct = output_correct.to(torch.float) * union_diff.to(torch.float)

    union_num = union_diff.sum(dtype=torch.float)
    intersection_num = masked_output_correct.sum(dtype=torch.float)
    return intersection_num / union_num


def compute_corrects_accuracy(output: torch.Tensor, target: torch.Tensor, padding_value):
    """
    计算完全正确的结果数
    最后取平均
    """
    # target is (b, d1, d2, ..., dn), output is (b, d1, d2, ..., dn)
    target = target.view(output.shape[0], -1)
    output = output.view(output.shape[0], -1)
    b, l = target.shape
    masks = target.eq(padding_value)
    # 修正padding元素
    output = output.clone()
    output[masks] = padding_value
    # target is (b)
    corrects = output.eq(target).sum(dim=1, dtype=torch.float)
    corrects = corrects.eq(l)
    return torch.true_divide( corrects.sum() , corrects.nelement())


def loging(writer, perfix, epoch, step, used_time, dataset_size, batch_size, tensorboard=True, **addentional):
    print(f"{perfix} epoch{epoch}", f"step{step}", f"samples {step % dataset_size}/{dataset_size}",
          f"net spend {format(used_time / batch_size, '.6f')}s",
          sep="    ", end="    ")

    for name in addentional:
        print(f"{perfix}_{name} {format(addentional[name], '.9f')}", end="    ")
        if writer is not None:
            writer.add_scalar(f"{perfix}_{name}", addentional[name], step) if tensorboard else None
    print("")
    if writer is not None:
        writer.flush() if tensorboard else None
