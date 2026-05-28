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
    <br>
    <a href="./README.md"><ins>English</ins></a> | 简体中文
  </p>
</div>

## 概览

**AlphaEarth4PUB** 是论文 [Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings](https://arxiv.org/abs/2601.01558) 的官方代码仓库。

本项目基于 Earth foundation models 生成的 AlphaEarth embeddings，提升水文模型预测能力。

## 环境配置

运行本项目需要从对应仓库安装以下依赖：

- **torchhydro**: [https://github.com/OuyangWenyu/torchhydro](https://github.com/OuyangWenyu/torchhydro)
- **hydrodataset**: [https://github.com/OuyangWenyu/hydrodataset](https://github.com/OuyangWenyu/hydrodataset)
- **hydroutils**: [https://github.com/OuyangWenyu/hydroutils](https://github.com/OuyangWenyu/hydroutils)
- **hydrodatasource**: [https://github.com/iHeadWater/hydrodatasource](https://github.com/iHeadWater/hydrodatasource)

建议在虚拟环境中安装这些依赖。可以克隆每个仓库后运行 `pip install -e .`，也可以参考各仓库的安装说明。

## 数据

本项目使用的 AlphaEarth 数据集可从 Zenodo 获取：
[https://zenodo.org/records/19159031](https://zenodo.org/records/19159031)

请下载数据，并按照配置文件要求组织数据目录结构。

## 使用方法

### 训练

运行以下命令训练模型：

```bash
python train.py
```

### 评估

运行以下命令评估训练好的模型：

```bash
python evaluate.py
```

## 许可证

本项目采用 MIT License，详细信息见 [LICENSE](LICENSE) 文件。

## 引用

如果您在研究中使用了本仓库或 AlphaEarth 水文建模流程，请引用：

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
