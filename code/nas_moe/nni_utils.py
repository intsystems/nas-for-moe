# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# type: ignore

"""File containing NASNet-series search space.

The implementation is based on NDS.
It's called ``nasnet.py`` simply because NASNet is the first to propose such structure.
"""

from functools import partial
from typing import Tuple, List, Union, Iterable, Dict, Callable, Optional, cast

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from torch import nn

import nni
from nni.mutable import MutableExpression, Sample
from nni.nas.nn.pytorch import ModelSpace, Repeat, Cell, MutableConv2d, MutableBatchNorm2d, MutableLinear, model_context

from nni.nas.oneshot.pytorch.supermodule.sampling import PathSamplingRepeat
from nni.nas.oneshot.pytorch.supermodule.differentiable import DifferentiableMixedRepeat


MaybeIntChoice = Union[int, MutableExpression]

# the following are NAS operations from
# https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/operations.py


# ==============================================================================
# ==============================================================================
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is copied from ``timm.layers.DropPath`` and modified to support both NASNet and AutoFormer,
    as well as the shape inference.

    Credit partially goes to `pt.darts <https://github.com/khanrc/pt.darts/blob/0.1/models/ops.py>`__.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    Drops some samples in the forward pass with probability ``drop_prob``.
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def _shape_forward(self, x):
        return x.real_shape

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    

from nni.common.blob_utils import NNI_BLOB, nni_cache_home, load_or_download_file


PRETRAINED_WEIGHT_URLS = {
    # proxylessnas
    'acenas-m1': f'{NNI_BLOB}/nashub/acenas-m1-e215f1b8.pth',
    'acenas-m2': f'{NNI_BLOB}/nashub/acenas-m2-a8ee9e8f.pth',
    'acenas-m3': f'{NNI_BLOB}/nashub/acenas-m3-66a5ed7b.pth',
    'proxyless-cpu': f'{NNI_BLOB}/nashub/proxyless-cpu-2df03430.pth',
    'proxyless-gpu': f'{NNI_BLOB}/nashub/proxyless-gpu-dbe6dd15.pth',
    'proxyless-mobile': f'{NNI_BLOB}/nashub/proxyless-mobile-8668a978.pth',

    # mobilenetv3
    'mobilenetv3-large-100': f'{NNI_BLOB}/nashub/mobilenetv3-large-100-420e040a.pth',
    'mobilenetv3-small-050': f'{NNI_BLOB}/nashub/mobilenetv3-small-050-05cb7a80.pth',
    'mobilenetv3-small-075': f'{NNI_BLOB}/nashub/mobilenetv3-small-075-c87d8acb.pth',
    'mobilenetv3-small-100': f'{NNI_BLOB}/nashub/mobilenetv3-small-100-8332faac.pth',
    'cream-014': f'{NNI_BLOB}/nashub/cream-014-060aea24.pth',
    'cream-043': f'{NNI_BLOB}/nashub/cream-043-bec949e1.pth',
    'cream-114': f'{NNI_BLOB}/nashub/cream-114-fc272590.pth',
    'cream-287': f'{NNI_BLOB}/nashub/cream-287-a0fcba33.pth',
    'cream-481': f'{NNI_BLOB}/nashub/cream-481-d85779b6.pth',
    'cream-604': f'{NNI_BLOB}/nashub/cream-604-9ee425f7.pth',

    # nasnet
    'darts-v2': f'{NNI_BLOB}/nashub/darts-v2-5465b0d2.pth',

    # spos
    'spos': f'{NNI_BLOB}/nashub/spos-0b17f6fc.pth',

    # autoformer subnet
    'autoformer-tiny-subnet': f'{NNI_BLOB}/nashub/autoformer-tiny-subnet-12ed42ff.pth',
    'autoformer-small-subnet': f'{NNI_BLOB}/nashub/autoformer-small-subnet-b4e25a1b.pth',
    'autoformer-base-subnet': f'{NNI_BLOB}/nashub/autoformer-base-subnet-85105f76.pth',
    # autoformer supernet
    'autoformer-tiny-supernet': f'{NNI_BLOB}/nashub/autoformer-tiny-supernet-6f107004.pth',
    'autoformer-small-supernet': f'{NNI_BLOB}/nashub/autoformer-small-supernet-8ed79e18.pth',
    'autoformer-base-supernet': f'{NNI_BLOB}/nashub/autoformer-base-supernet-0c6d6612.pth',
}


def load_pretrained_weight(name: str, **kwargs) -> str:
    if name not in PRETRAINED_WEIGHT_URLS:
        raise ValueError(f'"{name}" do not have a valid pretrained weight file.')
    url = PRETRAINED_WEIGHT_URLS[name]

    local_path = os.path.join(nni_cache_home(), 'nashub', url.split('/')[-1])
    load_or_download_file(local_path, url, **kwargs)
    return local_path

# ==============================================================================
# ==============================================================================

OPS = {
    'none': lambda C, stride, affine:
        Zero(stride),
    'avg_pool_2x2': lambda C, stride, affine:
        nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
    'avg_pool_3x3': lambda C, stride, affine:
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'avg_pool_5x5': lambda C, stride, affine:
        nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
    'max_pool_2x2': lambda C, stride, affine:
        nn.MaxPool2d(2, stride=stride, padding=0),
    'max_pool_3x3': lambda C, stride, affine:
        nn.MaxPool2d(3, stride=stride, padding=1),
    'max_pool_5x5': lambda C, stride, affine:
        nn.MaxPool2d(5, stride=stride, padding=2),
    'max_pool_7x7': lambda C, stride, affine:
        nn.MaxPool2d(7, stride=stride, padding=3),
    'skip_connect': lambda C, stride, affine:
        nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_1x1': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            MutableConv2d(C, C, 1, stride=stride, padding=0, bias=False),
            MutableBatchNorm2d(C, affine=affine)
        ),
    'conv_3x3': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            MutableConv2d(C, C, 3, stride=stride, padding=1, bias=False),
            MutableBatchNorm2d(C, affine=affine)
        ),
    'sep_conv_3x3': lambda C, stride, affine:
        SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine:
        SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine:
        SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine:
        DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine:
        DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_sep_conv_3x3': lambda C, stride, affine:
        DilSepConv(C, C, 3, stride, 2, 2, affine=affine),
    'conv_3x1_1x3': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            MutableConv2d(C, C, (1, 3), stride=(1, stride), padding=(0, 1), bias=False),
            MutableConv2d(C, C, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False),
            MutableBatchNorm2d(C, affine=affine)
        ),
    'conv_7x1_1x7': lambda C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            MutableConv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            MutableConv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            MutableBatchNorm2d(C, affine=affine)
        )
}

class ReLUConvBN(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_out, kernel_size, stride=stride,
                padding=padding, bias=False
            ),
            MutableBatchNorm2d(C_out, affine=affine)
        )


class DilConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_out, affine=affine),
        )


class SepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_out, affine=affine),
        )


class DilSepConv(nn.Sequential):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__(
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            MutableConv2d(
                C_in, C_in, kernel_size=kernel_size, stride=1,
                padding=padding, dilation=dilation, groups=C_in, bias=False
            ),
            MutableConv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            MutableBatchNorm2d(C_out, affine=affine),
        )


class Zero(nn.Module):

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super().__init__()
        if isinstance(C_out, int):
            assert C_out % 2 == 0
        else:   # is a value choice
            assert all(c % 2 == 0 for c in C_out.grid())
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = MutableConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = MutableConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = MutableBatchNorm2d(C_out, affine=affine)
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv_1(x), self.conv_2(y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class PixelPermutationLayer(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        # permutation — массив индексов для перестановки
        self.register_buffer('perm', permutation)

    def forward(self, x):
        # x.shape: (batch, channels, height, width)
        b, c, h, w = x.shape
        total_pixels = h * w
        assert total_pixels == len(self.perm), f"{total_pixels} {len(self.perm)}"

        x = x.view(b, c, -1)
        x = x[:, :, self.perm]
        x = x.view(b, c, h, w)
        return x



class AuxiliaryHead(nn.Module):
    def __init__(self, C: int, num_labels: int, dataset: Literal['imagenet', 'cifar', 'mnist']):
        super().__init__()
        if dataset == 'imagenet' or dataset == 'mnist':
            stride = 2 # input size 14x14 or 28x28
        elif dataset == 'cifar':
            stride = 3

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            MutableConv2d(C, 128, 1, bias=False),
            MutableBatchNorm2d(128),
            nn.ReLU(inplace=True),
            MutableConv2d(128, 768, 2, bias=False),
            MutableBatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = MutableLinear(768, num_labels)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class CellPreprocessor(nn.Module):
    """
    Aligning the shape of predecessors.

    If the last cell is a reduction cell, ``pre0`` should be ``FactorizedReduce`` instead of ``ReLUConvBN``.
    See :class:`CellBuilder` on how to calculate those channel numbers.
    """

    def __init__(self, C_pprev: MaybeIntChoice, C_prev: MaybeIntChoice, C: MaybeIntChoice, last_cell_reduce: bool) -> None:
        super().__init__()

        if last_cell_reduce:
            self.pre0 = FactorizedReduce(cast(int, C_pprev), cast(int, C))
        else:
            self.pre0 = ReLUConvBN(cast(int, C_pprev), cast(int, C), 1, 1, 0)
        self.pre1 = ReLUConvBN(cast(int, C_prev), cast(int, C), 1, 1, 0)

    def forward(self, cells):
        assert len(cells) == 2
        pprev, prev = cells
        pprev = self.pre0(pprev)
        prev = self.pre1(prev)

        return [pprev, prev]


class CellPostprocessor(nn.Module):
    """
    The cell outputs previous cell + this cell, so that cells can be directly chained.
    """

    def forward(self, this_cell, previous_cells):
        return [previous_cells[-1], this_cell]


class CellBuilder:
    """The cell builder is used in Repeat.
    Builds an cell each time it's "called".
    Note that the builder is ephemeral, it can only be called once for every index.
    """

    def __init__(self, op_candidates: List[str],
                 C_prev_in: MaybeIntChoice,
                 C_in: MaybeIntChoice,
                 C: MaybeIntChoice,
                 num_nodes: int,
                 merge_op: Literal['all', 'loose_end'],
                 first_cell_reduce: bool, last_cell_reduce: bool,
                 drop_path_prob: float):
        self.C_prev_in = C_prev_in      # This is the out channels of the cell before last cell.
        self.C_in = C_in                # This is the out channesl of last cell.
        self.C = C                      # This is NOT C_out of this stage, instead, C_out = C * len(cell.output_node_indices)
        self.op_candidates = op_candidates
        self.num_nodes = num_nodes
        self.merge_op: Literal['all', 'loose_end'] = merge_op
        self.first_cell_reduce = first_cell_reduce
        self.last_cell_reduce = last_cell_reduce
        self.drop_path_prob = drop_path_prob
        self._expect_idx = 0

        # It takes an index that is the index in the repeat.
        # Number of predecessors for each cell is fixed to 2.
        self.num_predecessors = 2

        # Number of ops per node is fixed to 2.
        self.num_ops_per_node = 2

    def op_factory(self, node_index: int, op_index: int, input_index: Optional[int], *,
                   op: str, channels: int, is_reduction_cell: bool):
        if is_reduction_cell and (
            input_index is None or input_index < self.num_predecessors
        ):  # could be none when constructing search space
            stride = 2
        else:
            stride = 1
        if is_reduction_cell and op == 'pixel_permutation':
            if stride == 1:
                return nn.Identity()
            else:
                return FactorizedReduce(channels, channels, affine=True)
        operation = OPS[op](channels, stride, True)
        if self.drop_path_prob > 0 and not isinstance(operation, nn.Identity):
            # Omit drop-path when operation is skip connect.
            # https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model.py#L54
            return nn.Sequential(operation, DropPath(self.drop_path_prob))
        return operation

    def __call__(self, repeat_idx: int):
        if self._expect_idx != repeat_idx:
            raise ValueError(f'Expect index {self._expect_idx}, found {repeat_idx}')

        # Reduction cell means stride = 2 and channel multiplied by 2.
        is_reduction_cell = repeat_idx == 0 and self.first_cell_reduce

        # self.C_prev_in, self.C_in, self.last_cell_reduce are updated after each cell is built.
        preprocessor = CellPreprocessor(self.C_prev_in, self.C_in, self.C, self.last_cell_reduce)

        ops_factory: Dict[str, Callable[[int, int, Optional[int]], nn.Module]] = {}
        for op in self.op_candidates:
            ops_factory[op] = partial(self.op_factory, op=op, channels=cast(int, self.C), is_reduction_cell=is_reduction_cell)

        cell = Cell(ops_factory, self.num_nodes, self.num_ops_per_node, self.num_predecessors, self.merge_op,
                    preprocessor=preprocessor, postprocessor=CellPostprocessor(),
                    label='reduce' if is_reduction_cell else 'normal')

        # update state
        self.C_prev_in = self.C_in
        self.C_in = self.C * len(cell.output_node_indices)
        self.last_cell_reduce = is_reduction_cell
        self._expect_idx += 1

        return cell

class NDSStage(Repeat):
    """This class defines NDSStage, a special type of Repeat, for isinstance check, and shape alignment.

    In NDS, we can't simply use Repeat to stack the blocks,
    because the output shape of each stacked block can be different.
    This is a problem for one-shot strategy because they assume every possible candidate
    should return values of the same shape.

    Therefore, we need :class:`NDSStagePathSampling` and :class:`NDSStageDifferentiable`
    to manually align the shapes -- specifically, to transform the first block in each stage.

    This is not required though, when depth is not changing, or the mutable depth causes no problem
    (e.g., when the minimum depth is large enough).

    .. attention::

       Assumption: Loose end is treated as all in ``merge_op`` (the case in one-shot),
       which enforces reduction cell and normal cells in the same stage to have the exact same output shape.
    """

    estimated_out_channels_prev: int
    """Output channels of cells in last stage."""

    estimated_out_channels: int
    """Output channels of this stage. It's **estimated** because it assumes ``all`` as ``merge_op``."""

    downsampling: bool
    """This stage has downsampling"""

    def first_cell_transformation_factory(self) -> Optional[nn.Module]:
        """To make the "previous cell" in first cell's output have the same shape as cells in this stage."""
        if self.downsampling:
            return FactorizedReduce(self.estimated_out_channels_prev, self.estimated_out_channels)
        elif self.estimated_out_channels_prev is not self.estimated_out_channels:
            # Can't use != here, ValueChoice doesn't support
            return ReLUConvBN(self.estimated_out_channels_prev, self.estimated_out_channels, 1, 1, 0)
        return None




_INIT_PARAMETER_DOCS = """

    Notes
    -----

    To use NDS spaces with one-shot strategies,
    especially when depth is mutating (i.e., ``num_cells`` is set to a tuple / list),
    please use :class:`~nni.nas.hub.pytorch.nasnet.NDSStagePathSampling` (with ENAS and RandomOneShot)
    and :class:`~nni.nas.hub.pytorch.nasnet.NDSStageDifferentiable` (with DARTS and Proxyless) into ``mutation_hooks``.
    This is because the output shape of each stacked block in :class:`~nni.nas.hub.pytorch.nasnet.NDSStage` can be different.
    For example::

        from nni.nas.hub.pytorch.nasnet import NDSStageDifferentiable
        darts_strategy = strategy.DARTS(mutation_hooks=[NDSStageDifferentiable.mutate])

    Parameters
    ----------
    width
        A fixed initial width or a tuple of widths to choose from.
    num_cells
        A fixed number of cells (depths) to stack, or a tuple of depths to choose from.
    dataset
        The essential differences are in "stem" cells, i.e., how they process the raw image input.
        Choosing "imagenet" means more downsampling at the beginning of the network.
    auxiliary_loss
        If true, another auxiliary classification head will produce the another prediction.
        This makes the output of network two logits in the training phase.
    drop_path_prob
        Apply drop path. Enabled when it's set to be greater than 0.

"""



class NDScustom(ModelSpace):
    __doc__ = """
    The unified version of NASNet search space.

    We follow the implementation in
    `unnas <https://github.com/facebookresearch/unnas/blob/main/pycls/models/nas/nas.py>`__.
    See `On Network Design Spaces for Visual Recognition <https://arxiv.org/abs/1905.13214>`__ for details.

    Different NAS papers usually differ in the way that they specify ``op_candidates`` and ``merge_op``.
    ``dataset`` here is to give a hint about input resolution, so as to create reasonable stem and auxiliary heads.

    NDS has a speciality that it has mutable depths/widths.
    This is implemented by accepting a list of int as ``num_cells`` / ``width``.
    """ + _INIT_PARAMETER_DOCS.rstrip() + """
    op_candidates
        List of operator candidates. Must be from ``OPS``.
    merge_op
        See :class:`~nni.nas.nn.pytorch.Cell`.
    num_nodes_per_cell
        See :class:`~nni.nas.nn.pytorch.Cell`.
    """

    def __init__(self,
                 op_candidates: List[str],
                 merge_op: Literal['all', 'loose_end'] = 'all',
                 num_nodes_per_cell: int = 4,
                 width: Union[Tuple[int, ...], int] = 16,
                 num_cells: Union[Tuple[int, ...], int] = 20,
                 dataset: Literal['cifar', 'imagenet', 'mnist'] = 'imagenet',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__()

        self.op_candidates = op_candidates
        self.merge_op = merge_op
        self.num_nodes_per_cell = num_nodes_per_cell
        self.width = width
        self.num_cells = num_cells
        self.dataset = dataset
        if dataset == 'cifar' or dataset == 'mnist':
            self.num_labels = 10
        elif dataset == 'imagenet':
            self.num_labels = 1000
        self.auxiliary_loss = auxiliary_loss
        self.drop_path_prob = drop_path_prob

        # preprocess the specified width and depth
        if isinstance(width, Iterable):
            C = nni.choice('width', list(width))
        else:
            C = width

        self.num_cells: MaybeIntChoice = cast(int, num_cells)
        if isinstance(num_cells, Iterable):
            self.num_cells = nni.choice('depth', list(num_cells))
        num_cells_per_stage = [(i + 1) * self.num_cells // 3 - i * self.num_cells // 3 for i in range(3)]

        # auxiliary head is different for network targetted at different datasets
        if dataset == 'imagenet':
            self.stem0 = nn.Sequential(
                MutableConv2d(3, cast(int, C // 2), kernel_size=3, stride=2, padding=1, bias=False),
                MutableBatchNorm2d(cast(int, C // 2)),
                nn.ReLU(inplace=True),
                MutableConv2d(cast(int, C // 2), cast(int, C), 3, stride=2, padding=1, bias=False),
                MutableBatchNorm2d(C),
            )
            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                MutableConv2d(cast(int, C), cast(int, C), 3, stride=2, padding=1, bias=False),
                MutableBatchNorm2d(C),
            )
            C_pprev = C_prev = C_curr = C
            last_cell_reduce = True
        elif dataset == 'cifar':
            self.stem = nn.Sequential(
                MutableConv2d(3, cast(int, 3 * C), 3, padding=1, bias=False),
                MutableBatchNorm2d(cast(int, 3 * C))
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False
        elif dataset == 'mnist':
            self.stem = nn.Sequential(
                MutableConv2d(1, cast(int, 3 * C), 3, padding=1, bias=False),         # <--- 1 канал
                MutableBatchNorm2d(cast(int, 3 * C)),
            )
            C_pprev = C_prev = 3 * C
            C_curr = C
            last_cell_reduce = False
        else:
            raise ValueError(f'Unsupported dataset: {dataset}')

        self.stages = nn.ModuleList()
        for stage_idx in range(3):
            if stage_idx > 0:
                C_curr *= 2
            # For a stage, we get C_in, C_curr, and C_out.
            # C_in is only used in the first cell.
            # C_curr is number of channels for each operator in current stage.
            # C_out is usually `C * num_nodes_per_cell` because of concat operator.
            cell_builder = CellBuilder(op_candidates, C_pprev, C_prev, C_curr, num_nodes_per_cell,
                                       merge_op, stage_idx > 0, last_cell_reduce, drop_path_prob)
            stage: Union[NDSStage, nn.Sequential] = NDSStage(cell_builder, num_cells_per_stage[stage_idx])

            if isinstance(stage, NDSStage):
                stage.estimated_out_channels_prev = cast(int, C_prev)
                stage.estimated_out_channels = cast(int, C_curr * num_nodes_per_cell)
                stage.downsampling = stage_idx > 0

            self.stages.append(stage)

            # NOTE: output_node_indices will be computed on-the-fly in trial code.
            # When constructing model space, it's just all the nodes in the cell,
            # which happens to be the case of one-shot supernet.

            # C_pprev is output channel number of last second cell among all the cells already built.
            if len(stage) > 1:
                # Contains more than one cell
                C_pprev = len(cast(Cell, stage[-2]).output_node_indices) * C_curr
            else:
                # Look up in the out channels of last stage.
                C_pprev = C_prev

            # This was originally,
            # C_prev = num_nodes_per_cell * C_curr.
            # but due to loose end, it becomes,
            C_prev = len(cast(Cell, stage[-1]).output_node_indices) * C_curr

            # Useful in aligning the pprev and prev cell.
            last_cell_reduce = cell_builder.last_cell_reduce

            if stage_idx == 2:
                C_to_auxiliary = C_prev

        if auxiliary_loss:
            assert isinstance(self.stages[2], nn.Sequential), 'Auxiliary loss can only be enabled in retrain mode.'
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, self.num_labels, dataset=self.dataset)  # type: ignore

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = MutableLinear(cast(int, C_prev), self.num_labels)

    def forward(self, inputs):
        if self.dataset == 'imagenet':
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
        else:
            s0 = s1 = self.stem(inputs)

        for stage_idx, stage in enumerate(self.stages):
            if stage_idx == 2 and self.auxiliary_loss and self.training:
                assert isinstance(stage, nn.Sequential), 'Auxiliary loss is only supported for fixed architecture.'
                for block_idx, block in enumerate(stage):
                    # auxiliary loss is attached to the first cell of the last stage.
                    s0, s1 = block([s0, s1])
                    if block_idx == 0:
                        logits_aux = self.auxiliary_head(s1)
            else:
                s0, s1 = stage([s0, s1])

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self.training and self.auxiliary_loss:
            return logits, logits_aux  # type: ignore
        else:
            return logits

    @classmethod
    def extra_oneshot_hooks(cls, strategy):
        from nni.nas.strategy import DARTS, RandomOneShot
        if isinstance(strategy, DARTS):
            return [NDSStageDifferentiable.mutate]
        elif isinstance(strategy, RandomOneShot):
            return [NDSStagePathSampling.mutate]
        return []


    def freeze(self, sample: Sample) -> None:
        """Freeze the model according to the sample.

        As different stages have dependencies among each other, we will recreate the whole model for simplicity.
        For weight inheritance purposes, this :meth:`freeze` might require re-writing.

        Parameters
        ----------
        sample
            The architecture dict.

        See Also
        --------
        nni.nas.nn.pytorch.MutableModule.freeze
        """
        with model_context(sample):
            return NDS(
                self.op_candidates,
                self.merge_op,
                self.num_nodes_per_cell,
                self.width,
                self.num_cells,
                self.dataset,
                self.auxiliary_loss,
                self.drop_path_prob
            )



    def set_drop_path_prob(self, drop_prob):
        """
        Set the drop probability of Drop-path in the network.
        Reference: `FractalNet: Ultra-Deep Neural Networks without Residuals <https://arxiv.org/pdf/1605.07648v4.pdf>`__.
        """
        for module in self.modules():
            if isinstance(module, DropPath):
                module.drop_prob = drop_prob

class DARTScustom(NDScustom):
    __doc__ = """Search space proposed in `Darts: Differentiable architecture search <https://arxiv.org/abs/1806.09055>`__.

    It is built upon :class:`~nni.nas.nn.pytorch.Cell`, and implemented based on :class:`~nni.nas.hub.pytorch.nasnet.NDS`.
    Its operator candidates are :attr:`~DARTS.DARTS_OPS`.
    It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.

    .. note::

        ``none`` is not included in the operator candidates.
        It has already been handled in the differentiable implementation of cell.

    """ + _INIT_PARAMETER_DOCS

    DARTS_OPS = [
        # 'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
    ]
    """The candidate operations."""

    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet', 'mnist'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.):
        super().__init__(self.DARTS_OPS,
                         merge_op='all',
                         num_nodes_per_cell=4,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)

    @classmethod
    def load_searched_model(
        cls, name: str,
        pretrained: bool = False, download: bool = False, progress: bool = True
    ) -> nn.Module:

        init_kwargs = {}  # all default

        if name == 'darts-v2':
            init_kwargs.update(
                num_cells=20,
                width=36,
            )
            arch = {
                'normal/op_2_0': 'sep_conv_3x3',
                'normal/op_2_1': 'sep_conv_3x3',
                'normal/input_2_0': [0],
                'normal/input_2_1': [1],
                'normal/op_3_0': 'sep_conv_3x3',
                'normal/op_3_1': 'sep_conv_3x3',
                'normal/input_3_0': [0],
                'normal/input_3_1': [1],
                'normal/op_4_0': 'sep_conv_3x3',
                'normal/op_4_1': 'skip_connect',
                'normal/input_4_0': [1],
                'normal/input_4_1': [0],
                'normal/op_5_0': 'skip_connect',
                'normal/op_5_1': 'dil_conv_3x3',
                'normal/input_5_0': [0],
                'normal/input_5_1': [2],
                'reduce/op_2_0': 'max_pool_3x3',
                'reduce/op_2_1': 'max_pool_3x3',
                'reduce/input_2_0': [0],
                'reduce/input_2_1': [1],
                'reduce/op_3_0': 'skip_connect',
                'reduce/op_3_1': 'max_pool_3x3',
                'reduce/input_3_0': [2],
                'reduce/input_3_1': [1],
                'reduce/op_4_0': 'max_pool_3x3',
                'reduce/op_4_1': 'skip_connect',
                'reduce/input_4_0': [0],
                'reduce/input_4_1': [2],
                'reduce/op_5_0': 'skip_connect',
                'reduce/op_5_1': 'max_pool_3x3',
                'reduce/input_5_0': [2],
                'reduce/input_5_1': [1]
            }

        else:
            raise ValueError(f'Unsupported architecture with name: {name}')

        model_factory = cls.frozen_factory(arch)
        model = model_factory(**init_kwargs)

        if pretrained:
            weight_file = load_pretrained_weight(name, download=download, progress=progress)
            pretrained_weights = torch.load(weight_file)
            model.load_state_dict(pretrained_weights)

        return model


class PixelPermutationSpace(NDScustom):
    __doc__ = """Search space with pixel shuffle operations.
    
    This search space includes:
    - Two convolution operations (3x3 and 5x5 separable convolutions)
    
    It is built upon :class:`~nni.nas.nn.pytorch.Cell`, and implemented based on :class:`~nni.nas.hub.pytorch.nasnet.NDS`.
    It has 4 nodes per cell, and the output is concatenation of all nodes in the cell.
    
    """ + _INIT_PARAMETER_DOCS
    
    PIXELPERMUTATION_OPS = [
        'sep_conv_3x3',      # Separable 3x3 convolution
        'sep_conv_5x5',      # Separable 5x5 convolution
        'pixel_permutation',     # Novel pixel rearrangement operation
    ]
    """The candidate operations with pixel permutation."""
    
    def __init__(self,
                 width: Union[Tuple[int, ...], int] = (16, 24, 32),
                 num_cells: Union[Tuple[int, ...], int] = (4, 8, 12, 16, 20),
                 dataset: Literal['cifar', 'imagenet', 'mnist'] = 'cifar',
                 auxiliary_loss: bool = False,
                 drop_path_prob: float = 0.,
                 permutation: Optional[torch.Tensor] = None):
        OPS['pixel_permutation'] = lambda C, stride, affine: PixelPermutationLayer(permutation)
        super().__init__(self.PIXELPERMUTATION_OPS,
                         merge_op='all',
                         num_nodes_per_cell=4,
                         width=width,
                         num_cells=num_cells,
                         dataset=dataset,
                         auxiliary_loss=auxiliary_loss,
                         drop_path_prob=drop_path_prob)

        
    