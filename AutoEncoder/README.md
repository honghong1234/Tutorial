# 自编码器
- 简介
	- 自编码通常被用于数据降维、特征提取等方面，是一种神经网络专有的无监督学习方法，一般与PCA（主成分分析）对比，通常认为比主成分分析这样的线性模型具有更好的效果。
	- 本案例只是介绍最基础的自编码并可视化降维、聚类效果，后续有时间再介绍CAE（卷积自编码器）。
- 步骤	
	- 数据获取
		- 依旧使用mnist手写集，**注意：普通网络对一般图片特征提取能力未必像手写这种简单数据效果这么好，对图片自编码建议使用CAE**。
		- 代码
			- ```python
				from keras.datasets import mnist
				
				(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 这里只需要使用x，为了对应接口才拿出y的
				
				x_train = x_train.astype('float32') / 255.  # minmax归一
				x_train.reshape(x_train.shape[0], -1)
				x_test = x_test.astype('float32') / 255.
				x_test.reshape(x_test.shape[0], -1)
				
				print(x_train.shape, x_test.shape)
				```
	- 模型构建
		- 这是一个很简单的全连接网络。
		- 代码
			- ```python
				# 建立模型
				from keras.layers import Input, Dense
				from keras.models import Model
				
				encoding_dim = 2
				
				img_size = Input(shape=(784, ))
				
				# 编码
				encoded = Dense(128, activation='relu')(img_size)
				encoded = Dense(64, activation='relu')(encoded)
				encoded = Dense(16, activation='relu')(encoded)
				encoder_output = Dense(encoding_dim)(encoded)
				
				# 解码
				decoded = Dense(16, activation='relu')(encoder_output)
				decoded = Dense(64, activation='relu')(decoded)
				decoded = Dense(128, activation='relu')(decoded)
				decoded = Dense(784, activation='tanh')(decoded)
				
				autoencoder = Model(inputs=img_size, outputs=decoded)
				encoder = Model(inputs=img_size, outputs=encoder_output)
				
				autoencoder.compile(optimizer='adam', loss='mse')
				
				autoencoder.summary()
				
				```
		- 结构可视化
			- ![](https://img-blog.csdnimg.cn/20190504133253558.png)
	- 模型训练
		- 自编码其实有两个过程，一个编码一个解码，所以它的x和y其实都是自己，当经过编码再解码之后获得的数据很接近原来的数据，那么取出编码模型，可以认为，编码器取得了最核心的数据。
		- 代码
			- ```python
				autoencoder.fit(x_train, x_train, batch_size=32, epochs=20, shuffle=True, verbose=True)
				
				encoded_imgs = encoder.predict(x_test)
				
				import matplotlib.pyplot as plt
				%matplotlib inline
				plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
				plt.colorbar()
				plt.show()
				
				```
		- 可视化编码效果
			- ![](https://img-blog.csdnimg.cn/20190504135627524.png)
			- 不同类型的压缩码，在空间分布是区分的，这说明编码确实提取到了最主要的成分。
- 补充说明
	- 本案例使用Keras框架，这是基于TensorFlow的一个上层框架，如果新手一开始不理解算图编程，Keras是个流行的选择。
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）