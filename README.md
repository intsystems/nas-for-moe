# Structure Aware Neural Architecture Search for Mixture of Experts

[![GitHub Contributors](https://img.shields.io/github/contributors/intsystems/nas-for-moe)](https://github.com/intsystems/nas-for-moe/graphs/contributors)

<table>
    <tr>
        <td align="left"> <b> Author </b> </td>
        <td> Petr Babkin </td>
    </tr>
    <tr>
        <td align="left"> <b> Advisor </b> </td>
        <td> Oleg Bakhteev, PhD </td>
    </tr>
</table>

## Assets

- [Code](code)
- [Paper overleaf](https://www.overleaf.com/5135847529jtnqzphbjwmk#b47970)
- [Paper pdf](paper/main.pdf)
- [Slides pdf](slides/main.pdf)

## Abstract

The Mixture-of-Experts (MoE) layer, a sparsely activated neural architecture controlled by a
routing mechanism, has recently achieved remarkable success across large-scale deep learning tasks. In
parallel, Neural Architecture Search (NAS) has emerged as a powerful methodology for automatically
discovering high-performing neural network. However, the application of NAS methods to MoE architectures
remains an underexplored research area. In this work, we propose an architecture search framework for MoE
models, which explicitly leverages the underlying cluster structure of the data. We evaluate the proposed
approach on computer vision benchmarks and demonstrate that it outperforms baseline MoE architectures
trained on the same datasets in terms of accuracy and computational efficiency.

## Citation

If you find our work helpful, please cite us.
```BibTeX
@article{babkin2025structure,
    title={Title},
    author={Petr Babkin, Oleg Bakhteev},
    year={2025}
}
```

## Licence

Our project is MIT licensed. See [LICENSE](LICENSE) for details.