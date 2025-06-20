# Multimodal Conditional Information Bottleneck for Generalizable AI-Generated Image Detection
Official implement of [Multimodal Conditional Information Bottleneck for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2505.15217).


<div align="center">
<h3>Haotian Qin<sup>1</sup>, Dongliang Chang<sup>1*</sup>, Yueying Gao<sup>1</sup>, Bingyao Yu<sup>2</sup>, Lei Chen<sup>2</sup>, Zhanyu Ma<sup>1</sup></h3>

<sup>1</sup>Beijing University of Posts and Telecommunications, Beijing, China  
<sup>2</sup>Tsinghua University, Beijing, China  

*Corresponding author.
</div>

## News
- **June 2025**: Released training code and GenImage caption annotations. The initial codebase is now available for public use, including scripts for training the model and annotations for the GenImage dataset.

## TODO
- **Validation Code**: Implement and release validation scripts to evaluate the model's performance on various datasets.
- **Inference Code**: Develop and share inference scripts for applying the trained model to new data for AI-generated image detection.
- **Plug-and-Play**: Add a modular, plug-and-play implementation of the MMCIB framework to facilitate easy integration into other networks. 

## Installation
To get started with the project, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/Ant0ny44/InfoFD.git
   ```
2. Install dependencies:
   ```bash
   conda env create -f environment.yml -n infoFD
   ```
3. Modify the `env.ini`.
    ```ini
    [WANDB]
    TOKEN=YOUR_WANDB_TOKEN_HERE
    ```
4. Change the data path in `configs/EP1.yml`:
   ```yml
    data:


    # This is used to specify the cache path for training/validation/test data. 
    # You can also directly provide the path to preprocessed training/validation/test data here.
    # When using cached training, if the specified file path does not exist, 
    # preprocessing will be performed first, and the resulting data will be stored in the corresponding cache path.
    # Note that for training data, captions corresponding to the images are required.
    
    train_root_cache:  TRAIN_ROOT_CACHE_PATH 
    val_root_cache: VAL_ROOT_CACHE_PATH
    test_root_cache: TEST_ROOT_CACHE_PATH

    train_root: GENIMAGE_TRAIN_PATH
    train_captions_path: ./data/genImage_train_captions.json
    val_root: GENIMAGE_VAL_PATH
    test_root: GENIMAGE_TEST_PATH


    prompts: True
    shuffle: True
    num_workers: 14
    batch_size:  512
    ...
   ```
## Usage
### Training
The training code is available in the `train.py` directory. To train the model, run:
```bash
bash scripts/train_EP1.sh
```
To run the statistic results, run:
```bash
bash scripts/train_EP1_stas.sh
```
See the `configs/` directory for model details.

### GenImage Training Caption Annotations
The GenImage training caption annotations are available in the `data/genImage_train_captions.json` directory. These annotations provide textual descriptions for the GenImage dataset, generating by [InternVL](https://github.com/OpenGVLab/InternVL).

## Contact
Thank you for your interest. We're currently finalizing the code organization. If you have any questions, please don't hesitate to reach out at ant0ny@163.com.

## Citation
If you use this code or the GenImage annotations in your research, please cite our work using the following BibTeX entry:

```bibtex
@article{qin2025multimodal,
  title={Multimodal Conditional Information Bottleneck for Generalizable AI-Generated Image Detection},
  author={Qin, Haotian and Chang, Dongliang and Gao, Yueying and Yu, Bingyao and Chen, Lei and Ma, Zhanyu},
  journal={arXiv preprint arXiv:2505.15217},
  year={2025}
}
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.