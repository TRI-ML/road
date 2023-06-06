## Recursive Octree Auto-Decoder (ROAD)

PyTorch implementation of the CoRL 2022 paper ["ROAD: Learning an Implicit Recursive Octree
Auto-Decoder to Efficiently Encode 3D Shapes"](https://zakharos.github.io/projects/road/).

![road.gif](media/road.gif)

### Installation

To set up the environment using conda, use the following commands:

```
conda create -n road python=3.10
conda activate road
```

Install Pytorch for your specific version of CUDA (11.6 in this example) as well as additional dependencies as provided
in _requirements.txt_.

```
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Training and Inference

To demonstrate the workflow of our pipeline, we include 3 mesh models from
the [HomebrewedDB dataset](https://campar.in.tum.de/personal/ilic/homebreweddb/index.html).
The _config.yaml_ file stores default parameters for training and evaluation and points to the provided 3 models.

To start **training**, run the following script, which will first generate and store training data given the provided
meshes and then train ROAD using the curriculum procedure.

```
python train.py --config configs/config.yaml
```

To **visualize** the trained model run:

```
python visualize.py --config configs/config.yaml
```

Additionally, one can provide the **lods** parameter specifying a desired output level of detail (LoD).

```
python visualize.py --config configs/config.yaml --lods 5
```

To **evaluate** the trained model, run the following script:

```
python evaluate.py --config configs/config.yaml
```

### Pre-trained Models

ROAD models pre-trained
on [Thingi32](https://ten-thousand-models.appspot.com/), [Google Scanned Objects (GSO)](https://goo.gle/scanned-objects),
and [AccuCities](https://www.accucities.com/) can be found here:

| Dataset  | # Objects | Latent size | Link                                                                                           |
|----------|-----------|-------------|------------------------------------------------------------------------------------------------|
| Thingi32 | 32        | 64          | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/road/thingi32_l64.zip)  |
| Thingi32 | 32        | 128         | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/road/thingi32_l128.zip) |
| GSO      | 128       | 512         | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/road/gso128_l512.zip)   |
| GSO      | 256       | 512         | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/road/gso256_l512.zip)   |
| City     | 1         | 512         | [model](https://s3.amazonaws.com/tri-ml-public.s3.amazonaws.com/github/road/city_l512.zip)     |

To visualize the pre-trained model, download it under the _pretrained_ folder and run:

```
python visualize.py --config pretrained/model/config.yaml
```

### Reference

```
@inproceedings{zakharov2022road,
    title={ROAD: Learning an Implicit Recursive Octree Auto-Decoder to Efficiently Encode 3D Shapes},
    author={Sergey Zakharov and Rares Ambrus and Katherine Liu and Adrien Gaidon},
    booktitle={Conference on Robot Learning (CoRL)},
    year={2022},
    url={https://arxiv.org/pdf/2212.06193.pdf}
    }
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License</a>.