# MLDS-Sprint-17-Examples

## Task 1 Simple CNN-GAN on Fashion-MNIST 

**Objective:** Build a small GAN that generates 28×28 Fashion-MNIST images (shirts, shoes, etc.).
**Dataset:** Fashion-MNIST (28×28 grayscale).
**Generator (suggestion):**

* Input: latent vector (size 64).
* Small MLP or lightweight ConvTranspose stack (two ConvTranspose layers) → output 28×28.
* Activation: ReLU for hidden, Tanh or Sigmoid for output.
  **Discriminator (suggestion):**
* Simple CNN with 2 conv layers → fully connected → sigmoid output.
* LeakyReLU activations.
  **Loss / Optimizer:** Binary cross-entropy; Adam (lr=0.0002, betas=(0.5,0.999)).
  **Training:** Alternate discriminator and generator updates for a few epochs (e.g., 10–20).
  **Evaluation / Deliverables:**
* Grid image `fashion_generated.png` with sampled generated images.
* Short notebook cell printing discriminator loss and generator loss per epoch.
  **Hints:** Normalize images to [−1,1] if using Tanh output. Use small batch size (64) for quick runs.

---

## Task 2 Conditional MLP-GAN on small digits 

**Objective:** Implement a conditional GAN that generates 28×28 MNIST digits conditioned on class label (0–9).
**Dataset:** MNIST.
**Generator (suggestion):**

* Input: noise vector (size 50) concatenated with a one-hot label vector (10 dims).
* 2–3 fully connected layers → output 28×28 (flattened) → reshape.
  **Discriminator (suggestion):**
* Input: image (flattened) concatenated with label embedding → MLP → sigmoid.
  **Loss / Optimizer:** BCE loss; Adam lr=0.0002.
  **Training:** Train for few epochs; sample generated images for a chosen label (e.g., all “7”) to verify conditioning.
  **Deliverables:**
* `cgan_samples_label7.png` showing several generated digits for class 7.
* Short note: how well does conditioning work? (one paragraph)
  **Hints:** Concatenate label to input or use label embedding; check that generated digit visually matches label.

---

## Task 3 Simple VAE on Iris 

**Objective:** Train a small Variational Autoencoder (VAE) to compress the Iris dataset into 2 latent dimensions and visualize the latent space.
**Dataset:** Iris (4 numeric features). Standardize features before training.
**Encoder / Decoder (suggestion):**

* Encoder: FC 4 → 16 → outputs `mu` and `log_var` (each size 2).
* Reparameterize `z = mu + eps * exp(0.5*log_var)`.
* Decoder: FC 2 → 16 → 4 (reconstructed features).
  **Loss / Optimizer:** MSE reconstruction + KL divergence; Adam lr=0.001.
  **Training:** Train 100–300 epochs (small dataset).
  **Evaluation / Deliverables:**
* Scatter plot of 2-D latent codes colored by species.
* Reconstruction MSE on the dataset.
  **Hints:** Keep latent dim = 2 so you can plot easily. Fit scaler on train and inverse transform reconstructions for MSE interpretation.

---

## Task 4 VAE for Wine 

**Objective:** Train a VAE on the Wine dataset (13 features) with latent dimension 3 to compress/reconstruct the data.
**Dataset:** Wine (sklearn). Standardize inputs.
**Architecture:** Encoder 13 → 32 → mu/log_var (size 3). Decoder symmetric.
**Loss / Optimizer:** MSE + KL; Adam lr=0.001.
**Deliverables:**

* Reconstruction MSE and a scatter of two selected latent dims colored by class.
* Save the trained model file `vae_wine.pth` (or similar).
  **Hints:** Use small batch size (16–32) and modest epochs (100). Track training losses.

---

## Task 5 Autoencoder for MNIST reconstruction (non-variational)

**Objective:** Build a basic autoencoder (not VAE) to compress MNIST to a small bottleneck and reconstruct images. This is simpler than a VAE but teaches encoding/decoding.
**Dataset:** MNIST.
**Architecture suggestion:**

* Encoder: 28×28 → FC 784 → 256 → 64 (bottleneck).
* Decoder: 64 → 256 → 784 → reshape 28×28.
  **Loss / Optimizer:** MSE or BCE for pixel reconstruction; Adam lr=0.001.
  **Training:** 10–20 epochs.
  **Deliverables:**
* Show a figure with original vs reconstructed images (first 10 test images).
* Report average reconstruction loss.
  **Hints:** Use sigmoid output and BCE if inputs in [0,1]; otherwise scale to [-1,1] and use MSE.

---

## Task 6 Denoising Autoencoder on Fashion-MNIST 

**Objective:** Train a denoising autoencoder that removes additive Gaussian noise from Fashion-MNIST images.
**Dataset:** Fashion-MNIST. Add Gaussian noise (e.g., std=0.3) to inputs during training; targets are original images.
**Architecture:** Small conv autoencoder (encoder conv→pool, decoder convtranspose).
**Loss / Optimizer:** MSE; Adam lr=0.001.
**Training:** Train until reconstructions are visually reasonable (10–20 epochs).
**Deliverables:**

* Figure showing noisy input → reconstructed output → original for several examples.
* Short note about observed denoising quality.
  **Hints:** Use clipping after adding noise to keep pixel values valid. Use small network so training is fast.
