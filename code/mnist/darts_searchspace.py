"""Описание DARTS-подобных операций (для mnist_sgem и mnist_random_pretrain).

Все операции stride=1 — редукция выполняется отдельной фиксированной cell.
"""

import torch.nn as nn


DARTS_OPS = [
    "skip_connect",
    "sep_conv_3x3",
    "sep_conv_5x5",
    "dil_conv_3x3",
    "dil_conv_5x5",
    "max_pool_3x3",
    "avg_pool_3x3",
    "zero",
]

# Токены графа: "input" (источник) и "add" (агрегатор) добавляются к op-токенам.
OPS_NEW = ["input"] + DARTS_OPS + ["add"]


class SepConv(nn.Module):
    def __init__(self, C: int, k: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, k, padding=k // 2, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C: int, k: int, dilation: int = 2):
        super().__init__()
        pad = (k - 1) * dilation // 2
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C, C, k, padding=pad, dilation=dilation, groups=C, bias=False),
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x):
        return self.op(x)


class PoolOp(nn.Module):
    def __init__(self, C: int, kind: str):
        super().__init__()
        if kind == "max":
            self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        else:
            self.pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.bn = nn.BatchNorm2d(C)

    def forward(self, x):
        return self.bn(self.pool(x))


class Zero(nn.Module):
    def forward(self, x):
        return x.mul(0.0)


class Identity(nn.Module):
    def forward(self, x):
        return x


def make_op(op_name: str, C: int) -> nn.Module:
    if op_name == "skip_connect":
        return Identity()
    if op_name == "sep_conv_3x3":
        return SepConv(C, 3)
    if op_name == "sep_conv_5x5":
        return SepConv(C, 5)
    if op_name == "dil_conv_3x3":
        return DilConv(C, 3, dilation=2)
    if op_name == "dil_conv_5x5":
        return DilConv(C, 5, dilation=2)
    if op_name == "max_pool_3x3":
        return PoolOp(C, "max")
    if op_name == "avg_pool_3x3":
        return PoolOp(C, "avg")
    if op_name == "zero":
        return Zero()
    raise ValueError(f"Unknown op: {op_name}")
