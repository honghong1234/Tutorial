# 简单的CNN实现
- 简介
	- 和之前的Pytorch一样，利用mnist手写数据集测试卷积神经网络特征提取能力。
- 步骤
	- 获取数据集
		- keras自带这类练手数据集。
		- 代码
			- ```python
				from keras.datasets import mnist
				import matplotlib.pyplot as plt
				%matplotlib inline
				
				(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
				print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
				x_train = x_train.reshape(-1, 28, 28) / 255
				x_valid = x_valid.reshape(-1, 28, 28) / 255
				# 可视化数据
				plt.figure(figsize=(12, 8))
				for i in range(10):
				    plt.subplot(2, 5, i+1)
				    plt.title("label:{}".format(y_train[i]))
				    plt.imshow(x_train[i], cmap='gray')
				plt.show()
				```
		- 演示效果
			- ![](https://img-blog.csdnimg.cn/20190502132447708.png)
	- 模型搭建
		- 主要利用两个卷积层提取参数（可以理解为缩小图片长宽，在高度上提取特征)
		- 代码
			- ```python
				# 构建模型
				from keras.models import Sequential
				from  keras.layers import Convolution2D, MaxPooling2D, Activation, Flatten, Dense
				from keras.optimizers import Adam
				
				model = Sequential()
				
				model.add(Convolution2D(
				    batch_input_shape=(None, 28, 28, 1), # 输入数据维度
				    filters=32,  # 卷积核数目
				    kernel_size=3,  # 卷积核大小
				    strides=1,  # 步长
				    padding='same',  # (3-1)/2
				    data_format='channels_last'  #  通道位置，注意keras和torch不同，一般通道在最后
				))  # 加入一个卷积层，输出(28, 28, 32)
				model.add(Activation('relu'))  # 加入激活函数
				model.add(MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_last',))  # 输出(14, 14, 32)
				
				model.add(Convolution2D(64, 3, strides=1, padding='same', data_format='channels_last'))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(2, 2, 'same', data_format='channels_last'))  # 输出(8, 8, 64)
				
				model.add(Flatten())
				model.add(Dense(1024))
				model.add(Activation('relu'))  # (1024)
				
				model.add(Dense(10))
				model.add(Activation('softmax'))  # (10) 这里是概率
				
				model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
				
				model.summary()
				```
		- 可视化模型结构
			- 可以看到，训练参数还是不少的
			- ![](https://img-blog.csdnimg.cn/20190502135007442.png)
	- 训练过程
		- 这里只训练少了轮次（防止过拟合）
		- 代码
			- ```python
				# 训练模型
				history = model.fit(x_train.reshape(-1, 28, 28, 1), y_train, batch_size=64, epochs=10, validation_split=0.2, shuffle=True, verbose=True)
				```
		- 训练可视化
			- ![](https://img-blog.csdnimg.cn/20190502140935188.png)
	- 验证集评价
		- 代码
			- ```python
				loss, accuracy = model.evaluate(x_valid.reshape(-1, 28, 28, 1), y_valid)
				print(loss, accuracy)
				
				result = model.predict(x_valid[:10].reshape(-1, 28, 28, 1))
				plt.figure(figsize=(12, 8))
				for i in range(10):
				    plt.subplot(2, 5, i+1)
				    plt.imshow(x_valid[i], cmap='gray')
				    plt.title("true:{}pred:{}".format(np.argmax(y_valid[i], axis=0), np.argmax(result[i], axis=0)))
				plt.show()
				```
		- 可视化结果
			- ![](https://img-blog.csdnimg.cn/20190502141142956.png)
			- 在真个验证集上准确率达到了0.9905，还是不错的。
- 补充说明
	- 本案例使用Keras框架，这是基于TensorFlow的一个上层框架，如果新手一开始不理解算图编程，Keras是个流行的选择。
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）