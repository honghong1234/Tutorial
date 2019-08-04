# 简单实现RNN
## 简介
- 本案例使用RNN对MNIST数据集进行分类。（只是演示使用步骤，RNN并不适合图片特征提取，而是在自然语言处理中使用广泛。）
## 步骤
- 导入数据集并onehot编码标签
	- 代码
		- ```python
			from keras.datasets import mnist
			from keras.utils import to_categorical
			import matplotlib.pyplot as plt
			import numpy as np
			%matplotlib inline
			
			(x_train, y_train), (x_valid, y_valid) = mnist.load_data()
			x_train.reshape(-1, 28, 28)
			y_train = to_categorical(y_train, num_classes=10)
			x_valid.reshape(-1, 28, 28)
			y_valid = to_categorical(y_valid, num_classes=10)
			
			print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
			
			plt.figure(figsize=(12, 8))
			for i in range(10):
				plt.subplot(2, 5, i+1)
				plt.imshow(x_train[i], cmap='gray')
				plt.title(np.argmax(y_train[i], axis=0))
			plt.show()
			```
	- 演示结果
		- 类似之前
		- ![](https://img-blog.csdnimg.cn/20190503124917559.png)
- 搭建网络结构
	- 可以理解为将一个图片一行一行输入网络，共进行28步
	- 代码
		- ```python
			from keras.layers import SimpleRNN, Activation, Dense
			from keras.models import Sequential
			
			model = Sequential()
			
			model.add(SimpleRNN(  # 使用keras封装的模块
				batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),  # 这里可以理解为一个图片28行数据一次送入一行（如果是动态步长，keras实现略显麻烦）
				output_dim=50,  # 输出为50
				unroll=True,
			))
			
			model.add(Dense(10))  # 将50映射到10的空间中
			model.add(Activation('softmax'))  # 将打分使用softmax激活为概率
			model.summary()
			```
	- 演示效果
		- ![](https://img-blog.csdnimg.cn/20190503125727174.png)
- 训练及测试集验证
	- 这里发现Simple RNN处理图片序列分类一般，改用了LSTM，具体见代码
	- 代码
		- ```python
			batch_index = 0
			batch_size = 50
			for step in range(5000):
				# shape为(batch_num, steps, inputs/outputs)
				x_batch = x_train[batch_index: batch_index+batch_size, :, :]  # 一次取64张图片
				y_batch = y_train[batch_index: batch_index+batch_size, :]
				loss = model.train_on_batch(x_batch, y_batch)
				batch_index += batch_size
				batch_index = 0 if batch_index >= x_train.shape[0] else batch_index  # 一旦取完了所有数据，batch索引清空
			
				if step % 500 == 0:
					cost, accuracy = model.evaluate(x_valid, y_valid, batch_size=y_valid.shape[0], verbose=False)  # 将测试数据全部送入
					print('valid cost: ', cost, 'valid accuracy: ', accuracy)
			
			```
	- 演示结果
		- ![](https://img-blog.csdnimg.cn/20190503133635462.png)
## 补充说明
- 本案例使用Keras框架，这是基于TensorFlow的一个上层框架，如果新手一开始不理解算图编程，Keras是个流行的选择。
- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。具体代码见[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Keras/RNNDemo)，欢迎star或者fork。
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。