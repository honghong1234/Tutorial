# Fastai进行模型训练
@[toc]
## 简介
- 此前，已经写了两篇大致上模仿Fastai使用方式的MNIST数据集上的博客，写的比较乱，这一篇会相对规整一些。
- 本案例讲述如何使用fastai的模型进行训练，具体讲解不同的训练方式及技巧。
## 说明
- fastai没有明显的模型这一概念，不同于Keras，它的主要训练基础是一个Learner对象（我理解为学习器），这个对象绑定了PyTorch模型、数据集（包括训练集、验证集、测试集）、优化器、损失函数等。所有的训练以及训练产生的结果的应用都是基于这个学习器展开的。（**这种聚合所有训练元素的方式使得代码很简约，但是是否实用因人而异。**）
- basic_train模块定义Learner这个类，它也使用了PyTorch封装的optimizer（优化器），它定义了基本的训练过程，体现在你调用fit或者其他的fit方法的变体时。
- callback模块定义Callback类以及CallbackHandler类。后者负责训练循环与callback方法的信息交互。（**使用过PyTorch应该很容易理解这个过程**）
- callbacks模块实现了一些常用的callback方法，它们都是基于Callback类的。一些回调方法用于调度超参数如callbacks.one_cycle,callbacks.lr_finder和callbacks.general_sched。也有一些回调方法允许特殊的训练方式，如callbacks.fp16(混合精度)和callbacks.rnn。而Recorder和callbacks.hooks用于保存训练过程中产生的内部数据。
- train通过使用这些callbacks回调去实现有用的帮助功能，如损失变化图像等。metrics模块包含了很多函数和类这些都是用来评价训练的结果的，简单指标均为函数，复杂指标均为Callback的子类。
## 前置步骤
- 获取数据集
	- 还是使用MNIST数据集作为测试数据集，只选取MNIST中的两个标签数据，模拟二分类(只包含3和7两个数字)。
	- 代码
		- ```python
			mnist_path = untar_data(URLs.MNIST_SAMPLE)  # 从网络下载精简版数据集并保存本地，返回路径
			data = ImageDataBunch.from_folder(mnist_path)
			
			data.show_batch(ds_type=DatasetType.Train, rows=3, figsize=(3, 3))
			
			```
	- 可视化数据集
		- ![](https://img-blog.csdnimg.cn/20190510151202732.png)
- 创建学习器
	- 任何一个学习器对象至少需要两个参数即数据集和模型，这里创建了一个最简单的cnn结构模型作为模型，数据集使用mnist简化集。
	- 代码
		- ```python
			model = simple_cnn((3, 16, 16, 2))
			learner = Learner(data=data, model=model)
			```
	- 这就是最简单，最基础的Learner结构。
- 训练过程（**注意，训练可以指定GPU，一般不指定会默认使用GPU（如果安装了CUDA）**）
	- 最简单的训练方法就是fit，fit方法至少需要epochs参数，即需要训练多少轮。
		- 代码
			- `learner.fit(1)`
		- 结果
			- fastai默认输出训练过程，但是由于没有指定任何metric，所以默认只输出了训练集和验证集的损失。
			- ![](https://img-blog.csdnimg.cn/20190510152534516.png)
	- 对Learner添加一些metric
		- 代码
			- ```python
				model = simple_cnn((3, 16, 16, 2))
				learner = Learner(data=data, model=model, metrics=[accuracy, AUROC(), error_rate])
				learner.fit(1)
				```
		- 结果
			- 可以看到，更清晰了解了模型的训练情况。
			- ![](https://img-blog.csdnimg.cn/20190510153204982.png)
	- 不妨使用一些回调
		- 通过回调，可以尽可能实现训练优化的方法，如单周期调度训练。
		- 代码
			- ```python
				model = simple_cnn((3, 16, 16, 2))
				learner = Learner(data=data, model=model, metrics=[accuracy, AUROC(), error_rate])
				callback = OneCycleScheduler(learn=learner, lr_max=0.01)
				learner.fit(1, callbacks=callback)
				```
		- 训练结果
			- ![](https://img-blog.csdnimg.cn/20190510153636257.png)
			- 不管哪个指标，这个结果都比之前的好了不少，callback确实优化了training过程。
	- 自动添加的回调
		- Recoder回调会自动添加到Learner中，它记录了训练过程很多信息。
		- 打印出recoder内容并绘制其中的学习率变化曲线。
			- ![](https://img-blog.csdnimg.cn/20190510154040695.png)
			- 一般训练学习率是不变的，这里因为加了OneCycle，这个优质的回调方法会自动调节学习率。
	- **其实，很多非常实用的回调已经被train封装**
		- 这样，是需要直接调用特定的训练函数就可以使用想要的回调优化了。
		- 如上面的OneCycleScheduler只需要调用`learner.fit_one_cycle(1)即可。这里的OneCycle回调使用了一些比较合适的默认参数。
	- **当然，如果你使用的是applications提到的任务如vision、text等，那么这些模块以及封装了很不错的学习器，并且提供了不少模型的结构和预训练的模型。**
		- 例如，可以轻松迁移一个resnet预训练的模型。
		- 代码
			- ```python
				learner = cnn_learner(data, models.resnet18, metrics=[accuracy, AUROC(), error_rate])
				learner.fit_one_cycle(1)
				```
		- 结果
			- ![](https://img-blog.csdnimg.cn/20190510155611208.png)
		- 近几年，迁移学习的强大，尤其在比赛中，就不用我多说了吧。
## 补充说明
- 本案例使用Fastai框架，这是基于PyTorch的一个上层框架，是2019年以来一个流行的选择，[官方文档地址](https://docs.fast.ai/)给出，目前没有中文文档。
- 本案例的说明部分均基于官方文档，如果英文好的可直接通过上面的链接阅读文档。具体代码见[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Fastai/Train)，欢迎star或者fork。（开发环境为Jupyter，运行在Colab上，GPU为16G的T4）
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。