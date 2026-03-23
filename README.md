# AlphaEarth4PUB

Enhancing hydrological model prediction using AlphaEarth embeddings derived from Earth foundation models.

## Environment Setup

To run this project, you need to install the following dependencies from their respective repositories:

- **torchhydro**: [https://github.com/OuyangWenyu/torchhydro](https://github.com/OuyangWenyu/torchhydro)
- **hydrodataset**: [https://github.com/OuyangWenyu/hydrodataset](https://github.com/OuyangWenyu/hydrodataset)
- **hydroutils**: [https://github.com/OuyangWenyu/hydroutils](https://github.com/OuyangWenyu/hydroutils)
- **hydrodatasource**: [https://github.com/iHeadWater/hydrodatasource](https://github.com/iHeadWater/hydrodatasource)

We recommend installing these in a virtual environment. You can install them by cloning each repository and running `pip install -e .` or by following the installation guides in each repository.

## Data

The AlphaEarth dataset used in this project can be found on Zenodo:
[https://zenodo.org/records/19159031](https://zenodo.org/records/19159031)

Please download the data and ensure it is structured appropriately as required by the configuration.

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
