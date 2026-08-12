# rir-encoder

Auxiliary code accompanying the paper:
- Sina Khanagha, Timo Gerkmann [*"Your U-Net Dereverberation Model is Secretly an RIR Encoder"*](https://arxiv.org/abs/2606.09557), Interspeech 2026, Sydney, Australia.

This repository contains a simple script to train the contrastive learning based Conformer or ResNet34 RIR encoders that is discussed in our paper.

## Installation

You can clone the repository and install the required dependencies with:
```bash
git clone https://github.com/sp-uhh/rir-encoder.git
cd rir-encoder
pip install -r requirements.txt
```

## Training

The current `data_module.py` operates based on the assumption that you are working with two separate directories, one for RIR .wav files and another one for anechoic speech samples. You can modify to your needs if you are working with a different setup.

After modifying the file paths in the training loop, you can use the `train.py` script for training the model. The code is rather simple and almost all parts such as backbones, logging, etc. are easily modifiable. 

## Checkpoints
Here is also a checkpoint for the Conformer model: [Google Drive](https://drive.google.com/file/d/1uJN6SXFdda5hPFPjiuzeQLoW1Qo7b3lA/view)

## Embedding extraction
After training, you can use the `embedding_extraction.py` template to extract your embeddings for further analysis.

## Citations / References
We kindly ask you to cite our papers in your publication when using any of our research or code:
TODO
