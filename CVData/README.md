# CV之数据读取及处理模块
@[toc]
## 简介
- 在我熟知的深度学习领域主要就是CV和NLP两种不同数据类型的问题，这两类问题处理的模型一般区别也比较大。
- 本案例详细解释Fastai模块中图片数据集的读入、预处理、增广等。
- 这部分详细内容参考官方文档[这一节](https://docs.fast.ai/vision.data.html)。
## 说明
- 关于CV数据处理的模块都在fastai.vision.data模块下。这个模块定义了数据集处理的Image对象以及Image对象的转换方法。**注意，如果想要使用fastai的训练机制（见我[之前博客](https://blog.csdn.net/zhouchen1998/article/details/90071837)），必须将数据集处理为DataBunch对象，这是必须条件，ndarray、Torch.Tensor均不可以直接使用**。当然，vision以及封装了一个专门处理图片的ImageDataBunch，它是DataBunch的子类。
- 由于规范，一般数据集都会遵循一定的格式，所以ImageDataBunch封装了一系列处理各种主流数据集的方法。
	- ImageDataBunch.from_folder()
		- 这是为了处理ImageNet格式的数据集。
	- ImageDataBunch.from_df()
		- 这是为了处理pandas的DataFrame为说明文件的数据集，这个DataFrame包含两列，一列为文件名（通常叫做id列），一列为标签名（可能是字符串标签、数值标签或者回归问题的一系列数值）。
	- ImageDataBunch.from_csv()
		- 这是处理csv文件的API，处理的文件内容格式与上面的df一致。
	- ImageDataBunch.from_lists()
		- 这里通常处理两个对象，一个对象为图片文件名列表，一个对象为对应的目标列表。
	- ImageDataBunch.from_name_func()
		- 这里通常处理两个对象，一个对象为图片文件名列表，一个对象为从文件名获取target或者label的函数。
	- ImageDataBunch.from_name_re()
		- 这里通常处理两个对象，一个对象为图片文件名列表，一个对象为从文件名获取target或者label的正则匹配模式。
	- **注意，上面第一个方法不会随机划分训练集和验证集（因为默认这种格式数据集的验证集应该在valid文件夹内给出），后五个方法会自动划分训练集和验证集。
- 当然，你也可以使用data_block定制自己的数据读取等方法。
## 步骤
- 获取数据集
	- **这里为了演示方便，仍然使用MNIST数据集子集**。
	- 这一步代码做了什么，我之前的博客已经说得很明白了。
	- 代码
		- ```python
			import os
			mnist_path = untar_data(URLs.MNIST_SAMPLE)
			print(mnist_path)
			print(os.listdir(mnist_path))
			!cat /root/.fastai/data/mnist_sample/labels.csv | head -n 10
			```
	- 运行结果
		- ![](https://img-blog.csdnimg.cn/20190511103417804.png)
		- **显然，这是一个imagenet格式数据集。里面csv文件是fastai加的，方便演示from_csv方法。**
- **数据读取及预处理**
	- 代码
		- ```python
			tfms = get_transforms(do_flip=False)
			data = ImageDataBunch.from_folder(mnist_path, ds_tfms=tfms, size=24)
			```
	- 在这短短的两行代码里面其实发生了很多事情。
		- 首先，通过vision模块封装的get_transforms方法获得了一个tuple，这个tuple存放很多的Transform对象。（可以print查看，主要是对数据集随机试验crop，flip等增广数据集能力）
		- 但是在get_transfroms的时候，加入参数do_flip=False，这是因为对手写数据集，不希望有随机的翻转，因为3转过来就不是3了。
		- 然后，按照imagenet的风格从数据存放路径载入数据集，第一个参数为数据集路径，第二个参数为转化器，第三个参数为目标图片的大小。
		- **同时，对于任何一个DataBunch对象都会创建一个train_dl和一个valid_dl，数据类型都是PyTorch的DataLoader。
- 数据展示
	- 对于DataBunch对象都有一个show_batch方法，用于展示数据集内容，对于ImageDataBunch，则展示的是图片。
	- 这个方法一般设置两个参数：rows，设定显示的图片行数和列数，figsize，设定显示区域的大小。
	- 例如
		- `data.show_batch(rows=3, figsize=(5, 5))`
		- ![](https://img-blog.csdnimg.cn/20190511105108521.png)
- 多标签数据集
	- ```python
		planet = untar_data(URLs.PLANET_SAMPLE)
		!ls /root/.fastai/data/planet_sample
		df = pd.read_csv(planet/'labels.csv')
		print(df.head())
		
		tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
		data = ImageDataBunch.from_csv(planet, folder='train', size=128, suffix='.jpg', label_delim=' ', ds_tfms=tfms)
		
		data.show_batch(rows=3, figsize=(9, 9))
		data.show_batch(rows=3, figsize=(9, 9),  ds_type=DatasetType.Valid)
		```
- 更多操作
	- 对于很多时候，其实ImageList是一个不错的选择，它的工厂方法是类似于ImageDataBunch的，它的读取对象一般针对图片数据。
		- 读取
			- 代码
				- ```python
					imagelistRGB = ImageList.from_folder(path_data/'train')
					print(imagelistRGB)
					imagelistRGB.items[10]
					imagelistRGB.open(imagelistRGB.items[10])
					```
			- 结果
				- ![](https://img-blog.csdnimg.cn/20190511111246218.png)
				- 存放的是很多Image对象，fastai的图片及显示是基于PIL的，可以设置mode，见GitHub代码。
	- fastai是一个有趣的框架，更多的功能去探索吧。
## 补充说明
- 本案例使用Fastai框架，这是基于PyTorch的一个上层框架，是2019年以来一个流行的选择，[官方文档地址](https://docs.fast.ai/)给出，目前没有中文文档。
- 本案例的说明部分均基于官方文档，如果英文好的可直接通过上面的链接阅读文档。具体代码见[我的Github](https://github.com/luanshiyinyang/DeepLearning/tree/Fastai/CVData)，欢迎star或者fork。（开发环境为Jupyter）
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。