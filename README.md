## Image classification using neural networks - Rabbit or Squirrel 
---
Two Jupyter notebooks with examples of building and training convolutional neural networks (CNNs) using **TensorFlow/Keras** and **PyTorch** tools.
### Overview:
---
The objective is to gain practical skills in building and training CNNs using TensorFlow/Keras and PyTorch tools. It deals with a task of binary classification.
The Jupyter notebooks describe the following stages of work: 
1) сleaning the used image dataset, viewing some of them with the matplotlib library, transforming images, for example, using augmentation;
3) building and training CNNs;
3) improvement of the built model with the help of normalization and regularization layers and hyperparameter tuning;
4) evaluation of better models on test data;
5) transfer learning with fine-tuning.

An idea was to describe all of the above stages using one deep learning tool, and then rewrite them to another. The first was created a notebook with the Keras library and on its basis a notebook using the Pytorch framework. Not everything has analogues. So there is no binary accuracy metric in Pytorch and we defined the function by analogy with the metric of the same name in TensorFlow/Keras. The same is true for hyperparameter search tools: in the first case, [**Keras Tuner**](https://github.com/keras-team/keras-tuner) was used, and in the second, [**Ray Tune**](https://docs.ray.io/en/latest/tune/index.html) (for more information, see [the paper](https://arxiv.org/abs/1807.05118)). The model used for transfer learning in the notebook with Pytorch is also different, but similar.
### Setup:
---
The Jupyter notebooks were created in the Google Colaboratory, so it's easier to open and run them there, but first, replace or recreate all the paths for loading and saving data. The data was stored on Google Drive in the following format:
```
image_store_dir/
    train/
        class1_name/
            image1.jpg
            image2.jpg
            ...
        class2_name/
            image3.jpg
            ...
    validation/
        ...
    test/
        ...
```
*The "validation" and "test" directories are organized in the same way as the "train" directory.*

***An example of downloading images from the Open Images in this format can be viewed [here](https://github.com/data42lana/download_images).***

The neural network models were trained using the GPU. To connect it, you need to select the GPU from the drop-down list in the Colab menu (Runtime-> Change runtime type) and save the selection.
### Versions of packages used:
---
python 3.7.10, tensorflow 2.5.0, keras 2.5.0, keras-tuner 1.0.3, torch 1.9.0+cu102, torchvision 0.10.0+cu102, ray 1.4.1, matplotlib 3.2.2.
### Data: 
---
Images of two classes "Rabbit" and "Squirrel" from [Open Images Dataset V6 + Extensions](https://storage.googleapis.com/openimages/web/index.html) licensed under a [Attribution 2.0 Generic (CC BY 2.0)](https://creativecommons.org/licenses/by/2.0/) license with a note (see the License section on the [Description page](https://storage.googleapis.com/openimages/web/factsfigures_v6.html) of the Dataset for more information).
