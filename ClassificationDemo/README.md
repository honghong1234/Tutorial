# 简单分类
## 简介
- 在这个案例中将从手写集开始了解fastai的使用模式。
## 步骤
- 获取数据集
	- fastai自带不少数据集，第一次使用需要下载。（本案例使用的是简化版数据集，只有3和7为二分类）
	- 代码
		- ```python
			mnist = untar_data(URLs.MNIST_TINY)  # 解压数据集并返回存放路径
			transform = get_transforms(do_flip=False)  # 得到一个翻转的转化器
			
			data = (ImageList.from_folder(mnist)
					.split_by_folder()          
					.label_from_folder()
					.add_test_folder('test')
					.transform(transform, size=32)
					.databunch()
					.normalize(imagenet_stats)) 
			print(type(data))
			```
	- 可以看到，通过这段代码，得到了一个fastai最常用的数据结构ImageDataBunch。
- 可视化数据集
	- 代码
		- ```python
			# 可视化数据
			# data.show_batch()  # 使用这种显示方法可能会溢出屏幕
			data.show_batch(rows=2, figsize=(2, 2))
			```
	- 可以看到，显示的效果还不错，想要更好看，就去修改参数。
		- ![](https://img-blog.csdnimg.cn/20190506205714738.png)
- 构建模型并训练
	- 代码
		- ```python
			# 创建模型
			learn = cnn_learner(data, models.resnet18, metrics=accuracy)  # 创建模型，使用预训练模型
			learn.fit(epochs=15, lr=0.01)
			learn.save('mnist_train')
			```
	- 演示结果
		- ![](https://img-blog.csdnimg.cn/20190506205856853.png)
		- 可以看到，fastai的训练过程类似Keras那样提供了一个不错的进度条和结果表格。
		- 15轮的训练，模型已经有些过拟合了；将模型保存在了本地。
- 利用模型进行预测分类
	- fastai对模型在新数据上的预测也提供了一个api。（总感觉将常用的都写好了）
	- 代码
		- ```python
			# learn.show_results()  # 这样显示屏幕也可能溢出
			learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(8,10))  # 这里指定数据分布类型为Train这样就不会显示有序数据而是打乱显示
			```
	- 结果
		- 分类的效果还是不错的。
		- ![](https://img-blog.csdnimg.cn/20190506210723886.png)
## 补充说明
- 本案例使用Fastai框架，这是基于PyTorch的一个上层框架，是2019年以来一个流行的选择，[官方文档地址](https://docs.fast.ai/)给出，目前没有中文文档。
- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。具体代码见[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Fastai/ClassificationDemo)，欢迎star或者fork。（开发环境为Jupyter，运行在Colab上，GPU为16G的T4）
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。