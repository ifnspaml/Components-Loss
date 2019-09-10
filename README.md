# Components Loss

These scripts are referring to the paper "Components Loss for Neural Networks in Mask-Based Speech Enhancement". In this repository, we provide the source code for training the mask-based speech enhancement convolutional neural networks (CNNs) using our proposed components loss (CL), which includes both 2 components loss (2CL) and 3 components loss (3CL). The corresponding test code is also offered.

The code was written by Ziyi Xu and with the help from Ziyue Zhao and Samy Elshamy.


## Introduction

We propose a novel components loss (CL) for the training of neural networks for mask-based speech enhancement. During the training process, the proposed CL offers separate control over preservation of the speech component quality, suppression of the residual noise component power, and preservation of a naturally sounding residual noise component. We obtain a better and more balanced performance in almost all employed instrumental quality metrics over the baseline losses, the latter comprising the conventional mean squared error (MSE) loss function and also auditory-related loss functions, such as the perceptual evaluation of speech quality (PESQ) loss and the recently proposed perceptual weighting filter loss.

## Prerequisites

- [Matlab](https://www.mathworks.com/) 2014a or later
- [Python](https://www.python.org/) 3.6
- CPU or NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-toolkit) 9.0 [CuDNN](https://developer.nvidia.com/cudnn) 7.0.5


## Getting Started

### Installation

- Install [TensorFlow](https://www.tensorflow.org/) 1.5.0 and [Keras](https://www.tensorflow.org/) 2.1.4
- Some Python packages need to be installed, please see detailed information in the Python scripts (e.g. numpy, scipy.io, and sklearn)
- Install [Matlab](https://www.mathworks.com/)

### Datasets

Note that in this project the clean speech signals are taken from the [Grid corpus](https://doi.org/10.1121/1.2229005) (downsampled to 16 kHz) and noise signals are taken from the [ChiMe-3](https://ieeexplore.ieee.org/abstract/document/7404837/) database.

### Training and validation data preparation

 - We use [Matlab](https://www.mathworks.com/) to prepare the input and target magnitude spectra for both training and validation sets.
 - To run the training script, you need:
1. ```training_input_noisy.mat``` (normalized noisy speech amplitude spectra, with zero mean and unit variance)
2. ```validation_input_noisy.mat``` (normalized noisy speech amplitude spectra, with zero mean and unit variance)
3. ```training_pure_noise.mat``` (amplitude spectra of noise component)
4. ```validation_pure_noise.mat``` (amplitude spectra of noise component)
5. ```training_clean_speech.mat``` (amplitude spectra of speech component)
6. ```validation_clean_speech.mat``` (amplitude spectra of speech component)
- All matrices have the dimensions of L*K (e.g. 1,000,000 *132). K represents the number of input and output frequency bins and is set to 132, and L represents the number of frames.
- All `.mat` files must be stored in `version 7.3`, using Matlab command `save('filename.mat','variable','-v7.3')` to enable very large data matrix saving.
- Small examples are placed under the directory: `./ training_data/`. To start your own training, replace these `.mat` files by your own data. More details are in the Python scripts. You can try the training script by using these small examples.

### Train the DNN models

 - Run the Python script to train the CNN model with the **proposed 2CL** based on the prepared training/validation data:
```bash
python Mask-based_CNN_2CL_training.py
```

 - Run the Python script to train the CNN model with the **proposed 3CL** based on the prepared training/validation data:
```bash
python Mask-based_CNN_3CL_training.py
```

### Test data preparation 

 - We also use [Matlab](https://www.mathworks.com/) to prepare the input magnitude spectra for test data and to store the phase information for the time-domain signal recovering.
- To run the test script, you need:
1. ```test_input_noisy_speech.mat``` (normalized noisy speech amplitude spectra, with zero mean and unit variance using the statistics collected on the training data)
2. ```test_pure_noise.mat``` (amplitude spectra of noise component, used to generate the _filtered_ noise component, which can be used for white-box based performance measures)
3. ```test_clean_speech.mat``` (amplitude spectra of speech component, used to generate the _filtered_ speech component, which can be used for white-box based performance measures)
4. ```test_noisy_speech_unmorm.mat``` (unnormalized noisy speech amplitude spectra, used for predicting enhanced speech)
- All matrices have the dimensions of L*K (e.g. 1,000 *132) as explained before.
- All `.mat` files are stored using Matlab command `save('filename.mat','variable')`, which allows to save maximum 2 GB `.mat` file. If you have a very large test data, you also need to store `.mat` files in `-v7.3`, and to modify the corresponding data loading part in the test script.
- Small examples are placed under the directory: `./ test_data /`. To start your test, replace these `.mat` files by your own data. More details are in the Python scripts. You can try the test script by using these small examples.
- The output of the scripts include:
1. ```test_n_tilde.mat.mat``` (_filtered_ noise amplitude spectra)
2. ```test_s_tilde.mat``` (_filtered_ speech amplitude spectra)
3. ```test_s_hat.mat``` (_enhanced_ speech amplitude spectra)
- The _filtered_ noise and speech amplitude spectra are then used to reconstruct the filtered noise and speech time domain signal, which can be used for white-box based performance measures.
### Test of the trained CNN models

 - Run the Python script to test the trained CNN model with the **proposed 2CL** using the prepared test data:
```bash
python Mask-based_CNN_2CL_test.py
```

 - Run the Python script to test the trained CNN model with the **proposed 3CL** using the prepared test data:
```bash
python Mask-based_CNN_3CL_test.py
```

### Time-domain signal reconstruction

 - The stored test data phase information is used to recover the time domain signal by IFFT with overlap add (OLA).
 
 ## Audio Demos
 - We provide audio demos using files from the test dataset in the presence of pedestrian (PED) noise at 10dB signal-to-noise ratio (SNR) level. The audios include speech from both female and male test speakers. 
 - We put the audio demos under the directory: `./Audio_demo/`.
 
 ## Citation

If you use the losses and/or scripts in your research, please cite

```
@article{xu2019Comploss,
  author =  {Z. Xu, S. Elshamy, Z. Zhao and T. Fingscheidt},
  title =   {{Components Loss for Neural Networks in Mask-Based Speech Enhancement}},
  journal = {arXiv preprint arXiv: 1908.05087},
  year =    {2019},
  month =   {Aug.}
}
```

## Acknowledgements
- The author would like to thank Ziyue Zhao and Samy Elshamy for the advice concerning the construction of these source code in GitHub.
