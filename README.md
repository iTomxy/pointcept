<div align="center">

<h1>An Empirical Study of End-to-End Pointcloud-based Rib Segmentation Model</h1>

<div>
  <a href='https://itomxy.github.io/' target='_blank'>Tianyou Liang</a><sup>1</sup>&emsp;
  <a href='https://scholar.google.com/citations?user=vzGWmWQAAAAJ&hl=en&inst=8615794581978883182' target='_blank'>Xiaoxu Li</a><sup>2</sup>&emsp;
  <a href='https://curvebeamai.com/about/our-team/dr-yu-peng-bs-phd/' target='_blank'>Yu Peng</a><sup>2</sup>&emsp;
  <a href='https://scholar.google.com.au/citations?user=Ac6VCMkAAAAJ&hl=en&inst=8615794581978883182' target='_blank'>Min Xu</a><sup>1</sup>&emsp;
</div>
<div>
  <sup>1</sup>University of Technology Sydney&emsp;
  <sup>2</sup>Curvebeam AI&emsp;
</div>

<div>
  <strong>Accepted to ICONIP 2026</strong><br/>
</div>

<div>
<h4 align="center">
  <!-- <a href="https://ieeexplore.ieee.org/document/10890755" target='_blank'>conference</a> • -->
  <a href="https://github.com/iTomxy/pointcept" target='_blank'>code (PTv3)</a> •
  <a href="https://github.com/iTomxy/openpoints-lighting" target='_blank'>code (PointCNN, PointNet, PointNet++, DGCNN, PointNeXt)</a>
  </h4>
</div>

<!-- <img src="asserts/framework.png" width="700px"/> -->

</div>

## Abstract
> *Automatic bone segmentation is a fundamental task supporting various clinical practices. Conventional methods in this field rely heavily on dense annotations, which incurs substantial labeling labor and expense. Recent efforts have been made to reduce the labeling workload through semi-supervised and weakly supervised learning. However, methods under these two paradigms usually assume that all objects of interest e.g., bones, are covered by labels. In this work, we explore a less studied problem setting that assumes only partially labeled bone CT data. To tackle the supervision bias brought by incomplete annotations, we design a three-stage learning method that automatically detects unlabeled bones while being robust to their various shape. Extensive experiments are conducted on the curated dataset to test the proposed method and promising performance is observed. To the best of our knowledge, this is the first work on the partially supervised bone segmentation problem.*

### Environment

Suggest using container (docker / singularity):

1. Base poincept environment:
use the built Docker image [by Pointcept](https://hub.docker.com/r/pointcept/pointcept/tags)
or [by us](https://hub.docker.com/repository/docker/tyloeng/pointcept/general)
(with [containers/pointcept-cu128_pt271.Dockerfile](containers/pointcept-cu128_pt271.Dockerfile)).

2. Patch with additional packages with
[containers/cu128_pt271.Dockerfile](containers/cu128_pt271.Dockerfile) (if using Docker)
or [containers/pointcept-cu128_pt271.def](containers/pointcept-cu128_pt271.def) (if using singularity).

### Datasets

See [data/ribsegv2.md](data/ribsegv2.md).

### Model training and evaluation

- [scripts/run-ribseg-ptv3.sh](scripts/run-ribseg-ptv3.sh): one-stage pipeline.
- [scripts/run-ribseg-ptv3-2stage.sh](scripts/run-ribseg-ptv3-2stage.sh): two-stage pipeline.

Configs:
- [configs/ribsegv2/seq-tune/](configs/ribsegv2/seq-tune/): sequential tuning.
- [configs/ribsegv2/semseg-pt_v3m1_0_base.py](configs/ribsegv2/semseg-pt_v3m1_0_base.py): tuned one-stage.
- [configs/ribsegv2/2-stage/](configs/ribsegv2/2-stage/): two-stage.

## Citation

COMING SOON

## License

[MIT License](LICENCE)

## Credit
This project is based on [Pointcept](https://github.com/Pointcept/Pointcept).

## Contact
If you have any questions, please feel free to reach out at `tianyou.liang@student.uts.edu.au`.
