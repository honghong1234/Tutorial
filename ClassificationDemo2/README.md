# Fastai模型的操作
- 简介
	- 本案例利用mnist手写数据集，详细介绍fastai中模型的生态。
- 步骤
	- 获取数据集
		- fastai自带很多数据集，mnist只是其中比较基础的。
		- 代码
			- ```python
				mnist = untar_data(URLs.MNIST_TINY)  # 该模块下载数据集到本地目录并返回路径
				tfms = get_transforms(do_flip=False)  # 创建转换器但是不进行翻转（注意不是所有数据集都适合任何模式的增广，对手写数字，翻转意味着变为另一个数字）
				# 利用ImageList从本地路径读入标准格式存放数据的数据集
				data = (ImageList.from_folder(mnist)
				        .split_by_folder()  # 按照数据集格式划分训练集测试集
				        .label_from_folder()  # 从数据集路径获取标签
				        .add_test_folder('test')  # 创建test目录
				        .transform(tfms, size=32)  # 利用构建的转换器的得到处理后的图片数据，大小调整为32*32
				        .databunch()  # 转为ImageDataBunch对象
				        .normalize(imagenet_stats))  # 使用imagenet标准规范化数据
				
				data.show_batch(rows=2, figsize=(4, 4))
				```
		- 可视化数据
			- ![](https://img-blog.csdnimg.cn/20190508202230101.png)
	- 构建模型
		- 这里提一下为什么不自己构建模型，类似keras那样，不断add构造基础模型？
			- 这并不是不行，只是fastai的理念不同于keras那样让模型构建更简单，它是让深度学习更便捷，而很多领域的神经网络结构已经基本成型如restnet、inception等，使用者在刚开始就可以便捷上手很多优质的模型。这就是为什么一行代码就创建了一个很强大的CNN结构。
		- 代码
			- `model = cnn_learner(data, models.resnet18, metrics=accuracy)`
	- 训练模型
		- 代码
			- `model.fit(epochs=10, lr=1e-3)`
		- 训练可视化
			- ![](https://img-blog.csdnimg.cn/20190508203001415.png)
	- 模型导出
		- 使用model.export()即可导出这个模型，不仅是模型结构。参数，包括数据集、数据集转换方式、回调等。
		- 导出的模型以export.pkl存在于model.path目录下，载入时只需要给出路径即可。（**这种存放方式很适合常见的工作目录就是数据集目录上层的方式**）
		- 具体过程如下图。
			- ![](https://img-blog.csdnimg.cn/20190508203949749.png)
	- 模型预测
		- 对单个数据进行预测，这里涉及到一个很关键的数据结构，也就是下图中的data.train_ds，可以这样理解，data就是数据集顶层文件夹，而里面的train数据集就是train_ds，这个train_ds包含很多信息（如图片的数目，每个图片的矩阵，每个图片的标签），利用下标索引一般第一个下标指第几个数据，第二个下标指x数据还是y数据。（**下图我就是取出前两个数据的x即特征数据进行预测**）
		- ![](https://img-blog.csdnimg.cn/20190508204735563.png)
		- 在这个图中，有一个有趣的现象，没有返回预测的类别，而是返回一格三元组，这个三元组包含三个属性，分别是**预测得到的对象（在这个例子就是标签）**、**潜在数据（在这个例子就是数据下标）**和**原始神经网络输出的概率**。
	- 在其他数据集预测
		- 其实这里的test数据还是读取时划分的，这个一般叫做验证集，但是实际项目、比赛时会给出一个真正的测试集，载入方式如下。
		- 代码
			- ```python
				# 载入模型，指定测试数据
				learner = load_learner(mnist, test=ImageList.from_folder(mnist/'test'))
				preds, y = learner.get_preds(ds_type=DatasetType.Test)  # 指定数据集类型为测试集
				print(preds[:10])
				print(y[:10])
				```
		- 运行结果
			- ![](https://img-blog.csdnimg.cn/20190508205851646.png)
- 补充说明
	- 本案例使用Fastai框架，这是基于PyTorch的一个上层框架，是2019年以来一个流行的选择，[官方文档地址](https://docs.fast.ai/)给出，目前没有中文文档。
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体代码见我的Github，欢迎star或者fork。（开发环境为Jupyter，运行在Colab上，GPU为16G的T4）