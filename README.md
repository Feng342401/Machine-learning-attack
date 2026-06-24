# Machine-learning-attack
Reproducible code for convolutional autoencoder-based PUF image generation and similarity evaluation against real cryptographic key images.

## Overview

This repository implements a machine learning-based adversarial evaluation framework for quantitatively assessing the robustness of image-based Physically Unclonable Function (PUF) key generation systems against data-driven inference attacks.

The core of the framework is a convolutional autoencoder, which is used as a surrogate attack model. Given a set of PUF images, the encoder compresses each sample into a latent representation, while the decoder reconstructs the original image space. The latent space is then regarded as an attack-sensitive feature manifold, and perturbations are introduced to simulate adversarial sampling around the learned representations. This process enables evaluation of whether statistically stable or structurally exploitable patterns exist within the PUF encoding space, and whether such patterns can be learned by deep neural networks.

To ensure a rigorous security evaluation protocol, the dataset is strictly divided into training and test subsets, with only the training data being accessible during model optimization. After training, the model is used to generate perturbed reconstructed images, which are then compared with both the training set and the unseen test set. Similarity evaluation is performed using pixel-wise mean squared error (MSE) and the structural similarity index (SSIM). Complete pairwise similarity matrices are generated to quantitatively characterize the degree of information leakage under machine learning-based attack conditions.

## Method Description

### Data and Splitting Strategy

- The dataset is organized into directories: training data are stored in `data/train/`, and test data are stored in `data/test/`, in PNG/JPG format.
- The code first reads each image and records its original width and height.
- During training, each image is resized such that the shorter side is scaled to a square dimension divisible by 8, followed by normalization to the range `[0,1]`.
- Only `data/train/` is used for training; only `data/test/` is used for evaluation. The test set tensors and filenames are saved to `output/test_tensors.pt`.

---

### Model Architecture

- A convolutional autoencoder is used:
  - Encoder: three convolutional layers with downsampling, followed by flattening and a fully connected layer to obtain a latent vector `z` (default dimension: 256).
  - Decoder: a fully connected layer reconstructs feature maps, followed by three transposed convolution layers for upsampling back to the training image resolution, and a final Sigmoid activation.
- The loss function is a weighted combination of MSE and L1 loss, emphasizing reconstruction fidelity to the original image.
- The learning rate is linearly decayed from an initial value `lr` to `1e-5` during training.

---

### Training Procedure

- Training input consists of images from the training set, with the same images used as reconstruction targets.
- At each epoch, the average training loss is computed and the current learning rate is reported.
- After training:
  - Model weights and image dimension information are saved to `output/autoencoder.pt`.
  - A subset of training images and their reconstructed counterparts are saved to `output/recon_samples`, with filenames such as `train_0_orig.png` and `train_0_recon.png` for visual inspection.

---

### Generation Procedure

- The trained model and image size information are loaded from `output/autoencoder.pt`.
- All images in `data/train/` are encoded into latent vectors `z` using the encoder.
- During generation:
  - A latent vector `z_base` is randomly selected from training samples.
  - Small Gaussian noise is added to `z_base` to obtain a perturbed latent vector `z`.
  - The decoder reconstructs the image from `z`, which is then resized back to the original image dimensions.

- Generated images are saved in `output/generated/` with filenames such as `gen_0000.png`.

---

### Matching Procedure

- **Generated vs Test Set**: Test tensors and filenames are loaded from `output/test_tensors.pt`. For each generated image and each test image, pixel-wise MSE is computed and saved to `output/match_results.csv`.
  - Additionally, in the final evaluation stage, SSIM is computed for all pairs between generated samples and test images, and the full SSIM matrix is saved to `output/match_results_ssim.csv`.

- **Generated vs Training Set**: For each generated image and each training image, pixel-wise MSE is computed and saved to `output/match_train_results.csv`.
  - Similarly, SSIM is computed for all pairs between generated samples and training images, and the full SSIM matrix is saved to `output/match_train_results_ssim.csv`.

(`match_results.csv` / `match_train_results.csv`: the first column corresponds to generated filenames, and remaining columns correspond to test/training images; lower MSE indicates higher similarity.)

(`match_results_ssim.csv` / `match_train_results_ssim.csv`: full SSIM matrices are provided, including values below the `--threshold` parameter.)

---

## Environment

Execute in the project root directory:

```bash
pip install -r requirements.txt
```
This project is released under the MIT License.



## Citation

If you use this software, please cite the associated publication when available.

## Usage

Script entry: train_and_generate.py

End-to-end pipeline (training + generation + matching)

```bash
python train_and_generate.py --mode all --epochs 200 --num_gen 1000 --threshold 0.8
```
Training only

```bash
python train_and_generate.py --mode train --epochs 200
```
Generation only (requires pretrained model)

```bash
python train_and_generate.py --mode gen --num_gen 1000
```
Matching evaluation only (requires existing test tensors and generated images)
```bash
python train_and_generate.py --mode match --threshold 0.8
```
During the matching stage, the threshold parameter does not affect SSIM matrix generation; all SSIM values are fully exported. If filtering is required, it can be applied post hoc on the exported CSV files.
