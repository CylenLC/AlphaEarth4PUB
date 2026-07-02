<div align="center">
  <h1 align="center"><sup>Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings</sup></h1>
  <p>
    <strong>
      Pengfei Qu, Wenyu Ouyang, Chi Zhang, Yikai Chai, Shuolong Xu, Lei Ye, Yongri Piao, Miao Zhang, Huchuan Lu
    </strong>
  </p>
  <p>
    <a href="https://arxiv.org/abs/2601.01558" style="text-decoration: none; margin: 0 8px;">
      <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white&labelColor=4d4d4d" alt="arXiv">
    </a>
    <a href="https://doi.org/10.5281/zenodo.21121620" style="text-decoration: none; margin: 0 8px;">
      <img src="https://zenodo.org/badge/1099816246.svg" alt="DOI">
    </a>
    <br>
    English | <a href="./README_zh.md"><ins>简体中文</ins></a>
  </p>
</div>

## Overview

**AlphaEarth4PUB** is the official repository for the paper [Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings](https://arxiv.org/abs/2601.01558).

This project enhances hydrological model prediction using AlphaEarth embeddings derived from Earth foundation models.

## Environment Setup

To run this project, install the following dependencies from their respective repositories:

- **torchhydro**: [https://github.com/OuyangWenyu/torchhydro](https://github.com/OuyangWenyu/torchhydro)
- **hydrodataset**: [https://github.com/OuyangWenyu/hydrodataset](https://github.com/OuyangWenyu/hydrodataset)
- **hydroutils**: [https://github.com/OuyangWenyu/hydroutils](https://github.com/OuyangWenyu/hydroutils)
- **hydrodatasource**: [https://github.com/iHeadWater/hydrodatasource](https://github.com/iHeadWater/hydrodatasource)

We recommend installing these dependencies in a virtual environment. You can clone each repository and run `pip install -e .`, or follow the installation guide in each repository.

## Data

The AlphaEarth dataset used in this project can be found on Zenodo:
[https://zenodo.org/records/19159031](https://zenodo.org/records/19159031)

Please download the data and ensure it is structured as required by the configuration.

## Usage

### Training

To train the model, run:

```bash
python train.py
```

### Evaluation

To evaluate the trained model, run:

```bash
python evaluate.py
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this repository or the AlphaEarth hydrological modeling workflow in your research, please cite:

```bibtex
@misc{qu2026utilizingearthfoundationmodels,
  title={Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings},
  author={Pengfei Qu and Wenyu Ouyang and Chi Zhang and Yikai Chai and Shuolong Xu and Lei Ye and Yongri Piao and Miao Zhang and Huchuan Lu},
  year={2026},
  eprint={2601.01558},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2601.01558},
  url={https://arxiv.org/abs/2601.01558}
}
```
