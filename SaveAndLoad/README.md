# 模型的保存与读取
## 简介
- 本项目使用简单的单层结构演示模型的保存与读取。
## 步骤
- 创建数据集。
	- 代码
		- ```python
			import numpy as np
			
			from keras.models import Sequential
			from keras.layers import Dense
			from keras.models import load_model
			from sklearn.model_selection import train_test_split
			
			x = np.linspace(-1, 1, 200)
			np.random.shuffle(x)
			y = 0.5 * x + 2 + np.random.normal(0, 0.05, (200, ))
			
			x_train, x_test,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2019)
			```
- 训练模型
	- 随便训练，效果不重要。
	- 代码
		- ```python
			model = Sequential()
			model.add(Dense(1, input_shape=(1,)))
			model.compile(loss='mse', optimizer='sgd')
			model.summary()
			
			model.fit(x_train, y_train, epochs=100, verbose=True)
			```
- 模型保存
	- 代码
		- ```python
			model = Sequential()
			model.add(Dense(1, input_shape=(1,)))
			model.compile(loss='mse', optimizer='sgd')
			model.summary()
			
			model.fit(x_train, y_train, epochs=100, verbose=True)
			```
	- 结果演示
		- ![](https://img-blog.csdnimg.cn/20190504142035420.png)
		- 可以看到，加载的模型与保存的是一致的。
- 模型参数保存
	- 很多时候大数据集上的模型是很大的，而且模型的保存是很频繁的，所以模型结构每次重新创建，只要载入参数即可。
	- 代码
		- ```python
			print("raw model in test", model.evaluate(x_test, y_test))
			model.save_weights('mymodel_weights.h5')
			
			# 新建模型，不训练
			model2 = Sequential()
			model2.add(Dense(1, input_shape=(1,)))
			model2.compile(loss='mse', optimizer='sgd')
			
			model2.load_weights('mymodel_weights.h5')
			print("load model weights in test", model2.evaluate(x_test, y_test))
			```
	- 结果演示
		- ![](https://img-blog.csdnimg.cn/20190504142101459.png)
		- 可以看到，加载权重的模型与保存的是一致的。
## 补充说明
- 本案例使用Keras框架，这是基于TensorFlow的一个上层框架，如果新手一开始不理解算图编程，Keras是个流行的选择。
- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。具体代码见[我的Github](https://github.com/luanshiyinyang/Tutorial/tree/Keras/SaveAndLoad)，欢迎star或者fork。
- 这是本系列最后一篇博客，后面会介绍最近比较火的，基于pytorch的顶层框架fastai。
- 博客同步至[个人博客网站](https://luanshiyinyang.github.io)，欢迎查看。