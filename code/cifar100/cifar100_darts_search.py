"""DARTS one-shot search в том же пространстве, что у SGEM.

Использует тот же `make_op` и те же топологические константы
(`NODE_EDGE_IDX`, `NODE_INPUT_CHOICES`), что и `cifar100_sgem.DartsCell`,
поэтому экспортированный config напрямую подаётся в существующий
`CIFAR100Net(config)`. Совместимость пространства проверяется sanity-чеком.

Этапы:
    1. Build supernet: stem(3→C) → cell1 → cell2 → reduction → GAP → FC,
       где cell1/cell2 — `DartsSearchCell` (LayerChoice + InputChoice) с
       одинаковыми label-ами (NNI шарит α между ячейками; веса операций
       независимы — как в SGEM).
    2. Веса W обучаются на train-выборке, архитектурные параметры α — на
       val-выборке (test нигде не используется на этапе поиска).
    3. NNI strategy.DARTS прогоняется N эпох (стандарт = 50).
    4. Экспортируется лучший sample → конвертация в SGEM-формат
       (op_0..op_8, input_0..input_8).
    5. Sanity-check: instantiate CIFAR100Net(config) и прогнать тестовый батч.
    6. Сохранить результат в JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

import nni  # noqa: F401
from nni.nas.nn.pytorch import LayerChoice, InputChoice, ModelSpace
from nni.nas.evaluator.pytorch import Lightning, ClassificationModule
from nni.nas.evaluator.pytorch.lightning import (
    Trainer as NniTrainer,
    DataLoader as NniDataLoader,
)
from nni.nas.experiment import NasExperiment
from nni.nas.strategy import DARTS

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import cifar100_searchspace  # noqa: E402
cifar100_searchspace.patch_toy_graph_ops()

from cifar100_searchspace import DARTS_OPS_SMALL, make_op  # noqa: E402
from cifar100_sgem import (  # noqa: E402
    NODE_EDGE_IDX, NODE_AGG_IDX, NODE_INPUT_CHOICES,
    FixedReductionCell, CIFAR100Net,
    load_cifar100_meta, CIFAR100DartsSearchSpace,
)
from cifar100_moe import load_cifar100_tensors  # noqa: E402


EDGE_TO_NODE = {0: 0, 1: 0, 3: 1, 4: 1, 6: 2, 7: 2}


# ==========================================================================
# Search-supernet (zeкало DartsCell, но с LayerChoice / InputChoice)
# ==========================================================================


class DartsSearchCell(nn.Module):
    """LayerChoice+InputChoice версия `cifar100_sgem.DartsCell`.

    Те же 6 op-рёбер ([0,1,3,4,6,7]) и те же кандидаты входов
    (NODE_INPUT_CHOICES). Каждое op-ребро → LayerChoice по
    DARTS_OPS_SMALL. Каждый input на ноде >0 → InputChoice(n_chosen=1).
    """

    def __init__(self, C: int):
        super().__init__()
        self.op_modules = nn.ModuleDict()
        self.input_choices = nn.ModuleDict()
        for edge_idx in [0, 1, 3, 4, 6, 7]:
            self.op_modules[str(edge_idx)] = LayerChoice(
                {op: make_op(op, C) for op in DARTS_OPS_SMALL},
                label=f"op_{edge_idx}",
            )
            node_i = EDGE_TO_NODE[edge_idx]
            n_choices = len(NODE_INPUT_CHOICES[node_i])
            if n_choices > 1:
                self.input_choices[str(edge_idx)] = InputChoice(
                    n_candidates=n_choices, n_chosen=1,
                    label=f"input_{edge_idx}",
                )
        self.out_proj = nn.Sequential(
            nn.Conv2d(3 * C, C, 1, bias=False),
            nn.BatchNorm2d(C),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cache = {-1: x}
        for node_i in (0, 1, 2):
            a_idx, b_idx = NODE_EDGE_IDX[node_i]
            choices = NODE_INPUT_CHOICES[node_i]
            outs = []
            for edge_idx in (a_idx, b_idx):
                if len(choices) == 1:
                    inp = cache[choices[0]]
                else:
                    inp = self.input_choices[str(edge_idx)](
                        [cache[c] for c in choices]
                    )
                outs.append(self.op_modules[str(edge_idx)](inp))
            cache[NODE_AGG_IDX[node_i]] = outs[0] + outs[1]
        concat = torch.cat([cache[2], cache[5], cache[8]], dim=1)
        return self.out_proj(concat)


class DartsSupernet(ModelSpace):
    """Supernet — мета-копия `cifar100_sgem.CIFAR100Net`.

    Cell1 и cell2 содержат отдельные LayerChoice/InputChoice инстансы с
    общими label-ами → NNI шарит α между ними (одна архитектурная
    конфигурация для двух ячеек, как в SGEM). Веса операций независимы.
    """

    def __init__(self, init_channels: int = 16, num_classes: int = 100):
        super().__init__()
        C = init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        self.cell1 = DartsSearchCell(C)
        self.cell2 = DartsSearchCell(C)
        self.reduction = FixedReductionCell(C)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.cell1(x)
        x = self.cell2(x)
        x = self.reduction(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ==========================================================================
# Конвертация NNI-export → SGEM config dict
# ==========================================================================


def nni_arch_to_sgem_config(arch: dict) -> dict:
    """Преобразовать `model.sample` от NNI → словарь, совместимый с DartsCell.

    NNI возвращает {label: choice}, где choice — ключ dict-кандидатов
    (для LayerChoice со словарём) или индекс (для InputChoice).
    """
    config: dict = {}
    for edge_idx in [0, 1, 3, 4, 6, 7]:
        v = arch[f"op_{edge_idx}"]
        if isinstance(v, int):
            v = DARTS_OPS_SMALL[v]
        if v not in DARTS_OPS_SMALL:
            raise ValueError(f"op_{edge_idx} export={v!r} не в {DARTS_OPS_SMALL}")
        config[f"op_{edge_idx}"] = v

    for edge_idx in [0, 1, 3, 4, 6, 7]:
        node_i = EDGE_TO_NODE[edge_idx]
        choices = NODE_INPUT_CHOICES[node_i]
        if len(choices) == 1:
            config[f"input_{edge_idx}"] = [choices[0]]
        else:
            v = arch[f"input_{edge_idx}"]
            if isinstance(v, list):
                v = v[0]
            idx = int(v)
            if not (0 <= idx < len(choices)):
                raise ValueError(
                    f"input_{edge_idx} export idx={idx} out of range "
                    f"for choices={choices}"
                )
            config[f"input_{edge_idx}"] = [choices[idx]]

    for node_i, agg_idx in NODE_AGG_IDX.items():
        a_idx, b_idx = NODE_EDGE_IDX[node_i]
        config[f"op_{agg_idx}"] = "add"
        config[f"input_{agg_idx}"] = [a_idx, b_idx]
    return config


# ==========================================================================
# Custom ClassificationModule с SGD+momentum+cosine (DARTS-style)
# ==========================================================================


class DartsClassificationModule(ClassificationModule):
    """ClassificationModule с SGD(momentum=0.9, nesterov)+CosineAnnealing."""

    def __init__(self, *, num_classes: int, learning_rate: float, weight_decay: float,
                 max_epochs: int):
        super().__init__(
            criterion=nn.CrossEntropyLoss,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer=torch.optim.SGD,
            num_classes=num_classes,
        )
        self._max_epochs = max_epochs

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
            nesterov=True,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self._max_epochs),
        )
        return [opt], [sched]


# ==========================================================================
# Main
# ==========================================================================


def main():
    ap = argparse.ArgumentParser(
        description="DARTS one-shot search на CIFAR-100 (SGEM-совместимое пространство)",
    )
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--save-results", type=str, required=True)
    ap.add_argument("--search-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-w", type=float, default=0.025)
    ap.add_argument("--lr-alpha", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=3e-4)
    ap.add_argument("--init-channels", type=int, default=16)
    ap.add_argument("--seed", type=int, default=322)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)

    data_dir = Path(args.data_dir)
    meta = load_cifar100_meta(data_dir)
    num_classes = int(meta["num_classes"])

    X_tr, y_tr, X_v, y_v = load_cifar100_tensors(data_dir)
    print(f"[data] num_classes={num_classes}, train={len(X_tr)}, val={len(X_v)}")
    print(f"[search] W (weights) ← train ({len(X_tr)}), "
          f"α (arch) ← val ({len(X_v)}); test не используется. "
          f"epochs={args.search_epochs}, batch_size={args.batch_size}")

    W_loader = NniDataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    A_loader = NniDataLoader(
        TensorDataset(X_v, y_v),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    supernet = DartsSupernet(
        init_channels=args.init_channels, num_classes=num_classes,
    )
    n_params = sum(p.numel() for p in supernet.parameters())
    print(f"[supernet] params: {n_params:,}")

    if args.device.startswith("cuda"):
        gpu_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        accelerator = "gpu"
        devices = [gpu_idx]
    else:
        accelerator = "cpu"
        devices = "auto"

    cls_module = DartsClassificationModule(
        num_classes=num_classes,
        learning_rate=args.lr_w,
        weight_decay=args.wd,
        max_epochs=args.search_epochs,
    )
    trainer = NniTrainer(
        max_epochs=args.search_epochs,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )
    evaluator = Lightning(
        cls_module, trainer,
        train_dataloaders=W_loader,
        val_dataloaders=A_loader,
    )
    strategy = DARTS(
        arc_learning_rate=args.lr_alpha,
        gradient_clip_val=5.0,
    )

    print(f"[strategy] DARTS arc_lr={args.lr_alpha}, grad_clip=5.0")
    exp = NasExperiment(supernet, evaluator, strategy)
    t0 = time.time()
    exp.run()
    elapsed = time.time() - t0
    print(f"[run] DARTS finished in {elapsed/60:.2f} min")

    samples = exp.export_top_models(top_k=1, formatter='dict')
    if not samples:
        raise RuntimeError("DARTS не вернул архитектур")
    arch = samples[0]
    print(f"[export] raw NNI sample:")
    for k in sorted(arch.keys()):
        print(f"   {k}: {arch[k]!r}")

    config = nni_arch_to_sgem_config(arch)
    print(f"[export] SGEM config:")
    for k in ["op_0", "op_1", "op_3", "op_4", "op_6", "op_7"]:
        print(f"   {k}: {config[k]}")
    for k in ["input_0", "input_1", "input_3", "input_4", "input_6", "input_7"]:
        print(f"   {k}: {config[k]}")

    # Sanity: convert to SGEM CIFAR100Net + forward
    expected_keys = set(
        CIFAR100DartsSearchSpace(init_channels=args.init_channels)
        .create_random_config().keys()
    )
    got_keys = set(config.keys())
    if expected_keys != got_keys:
        raise AssertionError(
            f"keys mismatch: expected={expected_keys}, got={got_keys}"
        )
    with torch.no_grad():
        net = CIFAR100Net(
            config, C=args.init_channels, num_classes=num_classes,
        )
        net.eval()
        x = torch.randn(2, 3, 32, 32)
        y = net(x)
        if y.shape != (2, num_classes):
            raise AssertionError(f"unexpected output shape: {y.shape}")
        n_p = sum(p.numel() for p in net.parameters())
        print(f"[sanity] CIFAR100Net forward OK, output={tuple(y.shape)}, "
              f"params={n_p:,}")

    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "cifar100_darts_search": {
            "config": config,
            "raw_arch": {k: (v if isinstance(v, (str, int, float)) else str(v))
                         for k, v in arch.items()},
            "search_epochs": args.search_epochs,
            "batch_size": args.batch_size,
            "lr_w": args.lr_w,
            "lr_alpha": args.lr_alpha,
            "wd": args.wd,
            "init_channels": args.init_channels,
            "data_dir": str(data_dir),
            "search_split": f"W=train({len(X_tr)})/A=val({len(X_v)})",
            "supernet_params": int(n_params),
            "time_sec": float(elapsed),
            "seed": args.seed,
            "device": args.device,
        }
    }
    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    tmp_path.replace(save_path)
    print(f"[save] -> {save_path}")


if __name__ == "__main__":
    main()
