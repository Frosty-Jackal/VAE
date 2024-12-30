# VAE-MNIST

## 项目简介

这个仓库包含了一个使用PyTorch实现的变分自编码器（Variational AutoEncoder, VAE）来生成MNIST手写数字的项目。VAE是一种生成模型，它能够在潜在空间中学习数据的分布，并从中生成新的数据样本。这个项目通过训练VAE模型，使其能够重建MNIST数据集中的手写数字，并生成新的手写数字图像。(本项目参照教程https://blog.csdn.net/zhouzongxin94/article/details/144760330)

## 项目结构

- `run.py`: 主脚本，包含数据加载、模型训练、损失计算、样本生成和模型保存等逻辑。
- `vae.py`: 定义VAE模型的脚本，包括编码器、解码器、重参数化技巧和前向传播方法。
- `./results/`: 保存生成的样本图像的目录。
- `./models/`: 保存训练好的VAE模型的目录。

## 依赖项

要运行这个项目，你需要安装以下Python库：

- `torch`: PyTorch深度学习框架。
- `torchvision`: 提供数据集加载和图像处理的工具。

你可以使用以下命令来安装这些依赖项：

```bash
pip install torch torchvision
```

## 数据集

这个项目使用MNIST数据集，它包含60,000个训练样本和10,000个测试样本，每个样本都是28x28像素的灰度手写数字图像。数据集会在第一次运行`run.py`时自动下载（如果`download=True`）。

## 运行项目

1. 克隆这个仓库到你的本地机器。
2. 导航到仓库的根目录。
3. 运行`run.py`脚本来训练VAE模型并生成样本图像。

```bash
python run.py
```

在运行过程中，训练损失会在每个epoch后打印出来，并且生成的样本图像会保存在`./results/`目录中，训练好的模型会保存在`./models/`目录中。

## 结果

- 在`./results/`目录中，你会看到每个epoch生成的样本图像，文件名格式为`sample_epoch_{epoch}.png`。
- 在`./models/`目录中，你会看到每个epoch保存的VAE模型，文件名格式为`vae_epoch_{epoch}.pth`。
- 在项目根目录下，你还会看到一个名为`generated_digits.png`的图像文件，它包含了从潜在空间中随机采样并解码生成的16个手写数字图像。

## 注意事项

- 你可以通过修改`run.py`中的参数来调整训练过程，比如更改batch size、学习率、训练epoch数等。
- 如果你没有GPU，代码会自动使用CPU进行训练，但训练速度可能会比较慢。
- 如果你想要加载之前保存的模型并继续训练或进行推理，可以使用`torch.load()`函数来加载模型。

## 贡献

如果你对这个项目有任何建议或贡献，欢迎在GitHub上提出issue或pull request。我们会非常感激你的帮助！

## 引用

如果你在你的研究或项目中使用了这个项目中的代码或数据，请引用这个GitHub仓库的链接。
