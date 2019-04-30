# 搭建简易分类网络
- 简介
	- 本案例使用一个隐层进行特征提取，一个隐层作为输出层进行MNIST手写数据集分类。
- 步骤
	- 获取数据集（keras自带常用练手数据集）
		- 代码
			- ```python
				# 导入MNIST数据集，该数据集Keras自带
				(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
				print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
				```
		- 可视化数据维度
			- ![](https://img-blog.csdnimg.cn/20190430124629194.png)
			- 可以看到，训练样本60,000个，验证样本10,000个，后面会将28*28展平为向量输入隐层1。

	- 数据预处理（标准化及标签编码）
		- 代码
			- ```python
				#  标准化数据
				x_train = x_train.reshape(x_train.shape[0], -1) / 255.  # 原数据是0-255不适合输入模型训练，标准化为0-1
				x_valid = x_valid.reshape(x_valid.shape[0], -1) / 255.
				# 标签one-hot编码，使用keras中工具
				y_train = np_utils.to_categorical(y_train, num_classes=10)
				y_test = np_utils.to_categorical(y_valid, num_classes=10)
				```
	- 模型构建
		- 代码
			- ```python
				# 模型搭建
				model = Sequential([
				    Dense(32, input_dim=28*28),
				    Activation('relu'),  # 使用relu激活
				    Dense(10),
				    Activation('softmax'),  # 使用softmax输出每个类别的概率
				])
				
				optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
				
				model.compile(optimizer=optimizer, loss='categorical_crossentropy',  # 分类多用交叉熵
				              metrics=['accuracy'])
				model.summary()
				```
		- keras结构显示
			- ![](https://img-blog.csdnimg.cn/20190430125506135.png)
	- 训练过程
		- 代码
			- ```python
				# 训练
				history = model.fit(x_train, y_train, epochs=100, verbose=False)
				import matplotlib.pyplot as plt
				%matplotlib inline
				plt.plot(np.arange(100), history.history['loss'])
				plt.show()
				```
			- 损失变化
				- 这里没有考虑过拟合，后面会提到，可以看到训练集上损失逐步降低。
				- ![](https://img-blog.csdnimg.cn/20190430130853250.png)
	- 预测测试集
		- 代码
			- ```python
				loss, accuracy = model.evaluate(x_valid, y_valid)
				print('test loss: ', loss)
				print('test accuracy: ', accuracy)
				
				pred = np.argmax(model.predict(x_valid[:10]), axis=1)
				print(pred)
				# 挑选部分输出
				plt.figure(figsize=(12, 8))
				for i in range(10):
				    plt.subplot(2, 5, i+1)
				    plt.imshow(x_valid[i].reshape(28, 28))
				    plt.title("True:{} Pred:{}".format(np.argmax(y_valid[i]), pred[i]))
				plt.show()
				```
		- 显示效果
			- 可以看到，准确率很高
			- ![](https://img-blog.csdnimg.cn/20190430131821663.png)
- 补充说明
	- 本案例使用Keras框架，这是基于TensorFlow的一个上层框架，如果新手一开始不理解算图编程，Keras是个流行的选择。
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）