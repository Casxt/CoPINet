# -*- coding: utf-8 -*-


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .blocks import *


class SimpleCoPINet(nn.Module):
    def __init__(self, num_attr=10, num_rule=6, channel_out=15, sample=False, dropout=False):
        super(SimpleCoPINet, self).__init__()
        self.num_attr = num_attr
        self.num_rule = num_rule
        self.sample = sample
        self.channel_out = channel_out

        self.internal_channel = 512

        self.conv1 = nn.Sequential(
            # nn.Conv2d(1, self.internal_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(1, self.internal_channel, kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(self.internal_channel, self.internal_channel, stride=1, downsample=nn.Sequential(
                conv1x1(self.internal_channel, self.internal_channel, stride=1),
                nn.BatchNorm2d(self.internal_channel)
            ))
        )
        self.bn1 = nn.BatchNorm2d(self.internal_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_row = conv3x3(self.internal_channel, self.internal_channel)
        self.bn_row = nn.BatchNorm2d(self.internal_channel, self.internal_channel)
        self.conv_col = conv3x3(self.internal_channel, self.internal_channel)
        self.bn_col = nn.BatchNorm2d(self.internal_channel, self.internal_channel)

        # self.inf_conv1 = nn.Conv2d(1, self.internal_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.inf_conv1 = nn.Sequential(
            # nn.Conv2d(1, self.internal_channel, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(1, self.internal_channel, kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(self.internal_channel, self.internal_channel, stride=1, downsample=nn.Sequential(
                conv1x1(self.internal_channel, self.internal_channel, stride=1),
                nn.BatchNorm2d(self.internal_channel)
            ))
        )
        self.inf_bn1 = nn.BatchNorm2d(self.internal_channel)

        self.inf_conv_row = conv3x3(self.internal_channel, self.internal_channel)
        self.inf_bn_row = nn.BatchNorm2d(self.internal_channel, self.internal_channel)
        self.inf_conv_col = conv3x3(self.internal_channel, self.internal_channel)
        self.inf_bn_col = nn.BatchNorm2d(self.internal_channel, self.internal_channel)

        self.predict_rule = nn.Linear(self.internal_channel, self.num_attr * self.num_rule)
        if self.sample:
            self.inference = GumbelSoftmax(temperature=0.5)
        else:
            self.inference = nn.Softmax(dim=-1)

        # basis
        self.basis_bias = nn.Linear(self.num_rule, self.internal_channel, bias=False)
        self.contrast1_bias_trans = MLP(in_dim=self.internal_channel,
                                        out_dim=self.internal_channel)  # nn.Linear(64, 64)
        self.contrast2_bias_trans = MLP(in_dim=self.internal_channel,
                                        out_dim=self.internal_channel)  # nn.Linear(64, 64)

        self.res1_contrast = conv3x3(self.internal_channel + self.internal_channel, self.internal_channel)
        self.res1_contrast_bn = nn.BatchNorm2d(self.internal_channel)
        self.res1 = ResBlock(self.internal_channel, self.internal_channel * 4, stride=2, downsample=nn.Sequential(
            conv1x1(self.internal_channel, self.internal_channel * 4, stride=2),
            nn.BatchNorm2d(self.internal_channel * 4)
        ))
        #
        # self.res2_contrast = conv3x3(128 + 64, 128)
        # self.res2_contrast_bn = nn.BatchNorm2d(128)
        # self.res2 = ResBlock(128, 256, stride=2, downsample=nn.Sequential(
        #     conv1x1(128, 256, stride=2),
        #     nn.BatchNorm2d(256)
        # ))

        # self.res3 = ResBlock(1024, 2048, stride=1, downsample=nn.Sequential(
        #         conv1x1(1024, 2048, stride=1),
        #         nn.BatchNorm2d(2048)
        #     ))
        self.res3 = nn.Sequential(
            ResBlock(self.internal_channel * 4, self.internal_channel * 4, stride=1, downsample=nn.Sequential(
                conv1x1(self.internal_channel * 4, self.internal_channel * 4, stride=1),
                nn.BatchNorm2d(self.internal_channel * 4)
            )),
            ResBlock(self.internal_channel * 4, self.internal_channel * 4, stride=1, downsample=nn.Sequential(
                conv1x1(self.internal_channel * 4, self.internal_channel * 4, stride=1),
                nn.BatchNorm2d(self.internal_channel * 4)
            ))
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = MLP(in_dim=self.internal_channel * 4, out_dim=channel_out, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 4, 40, 40)
        N, _, H, W = x.shape

        # Inference Branch
        prior = x[:, :3, :, :]
        input_features = self.maxpool(
            self.relu(self.inf_bn1(self.inf_conv1(prior.contiguous().view(-1, H, W).unsqueeze(1)))))
        input_features = input_features.view(-1, 3, self.internal_channel, H, W)

        row1_features = torch.sum(input_features[:, 0:2, :, :, :], dim=1)
        # row2_features = torch.sum(input_features[:, 3:6, :, :, :], dim=1)
        row_features = self.relu(self.inf_bn_row(self.inf_conv_row(row1_features)))
        final_row_features = row_features  # row_features[:N, :, :, :] + row_features[N:, :, :, :]

        # col1_features = torch.sum(input_features[:, 0:9:3, :, :, :], dim=1)
        # col2_features = torch.sum(input_features[:, 1:9:3, :, :, :], dim=1)
        # col_features = self.relu(self.inf_bn_col(self.inf_conv_col(torch.cat((col1_features, col2_features), dim=0))))
        # final_col_features = col_features[:N, :, :, :] + col_features[N:, :, :, :]

        input_features = final_row_features  # + final_col_features
        input_features = self.avgpool(input_features).view(-1, self.internal_channel)

        predict_rules = self.predict_rule(input_features)  # N, self.num_attr * self.num_rule
        predict_rules = predict_rules.view(-1, self.num_rule)
        predict_rules = self.inference(predict_rules)

        basis_bias = self.basis_bias(predict_rules)  # N * self.num_attr, 64
        basis_bias = torch.sum(basis_bias.view(-1, self.num_attr, self.internal_channel), dim=1)  # N, 64

        contrast1_bias = self.contrast1_bias_trans(basis_bias)
        contrast1_bias = contrast1_bias.view(-1, self.internal_channel, 1, 1).expand(-1, -1, H, W)
        contrast2_bias = self.contrast2_bias_trans(basis_bias)
        contrast2_bias = contrast2_bias.view(-1, self.internal_channel, 1, 1).expand(-1, -1, H // 2, W // 2)

        # Perception Branch
        input_features = self.maxpool(self.relu(self.bn1(self.conv1(x.view(-1, H, W).unsqueeze(1)))))
        input_features = input_features.view(-1, 4, self.internal_channel, H, W)

        choices_features = input_features[:, 3:, :, :, :].unsqueeze(2)  # N, 4, 64, 20, 20 -> N, 1, 1, 64, 20, 20

        row1_features = torch.sum(input_features[:, 0:2, :, :, :], dim=1)  # N, 64, 20, 20
        # row2_features = torch.sum(input_features[:, 3:6, :, :, :], dim=1)  # N, 64, 20, 20
        row3_pre = input_features[:, 2:3, :, :, :].unsqueeze(
            1)  # .expand(N, 1, 1, 64, 10,10) N, 1, 64, 10, 10 -> N, 1, 2, 64, 10, 10 -> N, 1, 1, 64, 10, 10

        row3_features = torch.sum(torch.cat((row3_pre, choices_features), dim=2), dim=2).view(-1, self.internal_channel,
                                                                                              H, W)  # N, 1, 1, 64, 10, 10 -> N, 1, 64, 10, 10 -> N * 1, 64, 10, 10
        row_features = self.relu(
            self.bn_row(self.conv_row(torch.cat((row1_features, row3_features), dim=0))))

        row1 = row_features[:N, :, :, :].view(N, 1, 1, self.internal_channel, H, W) # .unsqueeze(1).unsqueeze(1)#.expand(N, 1, 1, self.internal_channel, H, W)
        # row2 = row_features[N:2 * N, :, :, :].unsqueeze(1).unsqueeze(1).expand(N, 8, 1, 64, 20, 20)
        row3 = row_features[N:, :, :, :].view(N, 1, 1, self.internal_channel, H, W)# .unsqueeze(2)
        # print("row_features", row_features.shape)
        # print("row1", row1.shape)
        # print("row3", row3.shape)
        final_row_features = torch.sum(torch.cat((row1, row3), dim=2), dim=2)

        # col1_features = torch.sum(input_features[:, 0:9:3, :, :, :], dim=1)  # N, 64, 20, 20
        # col2_features = torch.sum(input_features[:, 1:9:3, :, :, :], dim=1)  # N, 64, 20, 20
        # col3_pre = input_features[:, 2:8:3, :, :, :].unsqueeze(1).expand(N, 8, 2, 64, 20,
        #                                                                  20)  # N, 2, 64, 20, 20 -> N, 1, 2, 64, 20, 20 -> N, 8, 2, 64, 20, 20
        # col3_features = torch.sum(torch.cat((col3_pre, choices_features), dim=2), dim=2).view(-1, 64, 20,
        #                                                                                       20)  # N, 8, 3, 64, 20, 20 -> N, 8, 64, 20, 20 -> N * 8, 64, 20, 20
        # col_features = self.relu(
        #     self.bn_col(self.conv_col(torch.cat((col1_features, col2_features, col3_features), dim=0))))
        #
        # col1 = col_features[:N, :, :, :].unsqueeze(1).unsqueeze(1).expand(N, 8, 1, 64, 20, 20)
        # col2 = col_features[N:2 * N, :, :, :].unsqueeze(1).unsqueeze(1).expand(N, 8, 1, 64, 20, 20)
        # col3 = col_features[2 * N:, :, :, :].view(-1, 8, 64, 20, 20).unsqueeze(2)
        # final_col_features = torch.sum(torch.cat((col1, col2, col3), dim=2), dim=2)

        input_features = final_row_features  # + final_col_features
        input_features = input_features.view(-1, self.internal_channel,  H, W)

        res1_in = input_features.view(-1, 1, self.internal_channel,  H, W)
        res1_contrast = self.res1_contrast_bn(
            self.res1_contrast(torch.cat((torch.sum(res1_in, dim=1), contrast1_bias), dim=1)))
        res1_in = res1_in - res1_contrast.unsqueeze(1)
        res2_in = self.res1(res1_in.view(-1, self.internal_channel, H // 2, W // 2))
        res2_in = res2_in.view(-1, 1, self.internal_channel + self.internal_channel,  H // 2, W // 2)
        res2_contrast = self.res2_contrast_bn(
            self.res2_contrast(torch.cat((torch.sum(res2_in, dim=1), contrast2_bias), dim=1)))
        res2_in = res2_in - res2_contrast.unsqueeze(1)
        res3_in = torch.nn.functional.interpolate(res2_in.view(-1, self.internal_channel + self.internal_channel, H // 2, W // 2), scale_factor=2, mode='nearest',
                                                  align_corners=None)
        res3_out = self.res3(res3_in.view(-1, self.internal_channel + self.internal_channel, H, W))
        final = res3_out.permute([0, 2, 3, 1]).contiguous()
        final = self.mlp(final).view(-1, H, W, self.channel_out)
        final = torch.nn.functional.log_softmax(final, dim=-1)
        return final

        # out = self.res2(res2_in.view(-1, 128, 10, 10))
        #
        # avgpool = self.avgpool(out)
        # avgpool = avgpool.view(-1, 256)
        # final = avgpool
        # final = self.mlp(final)
        # return final.view(-1, 8)
