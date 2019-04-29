# 搭建简易回归网络
- 简介
	- 本案例使用Keras搭建一个单隐层，20个神经元的神经网络进行回归分析。
- 步骤
	- 创建数据集
		- 代码
			- ```python
				# 创建数据集
				x = np.linspace(-1, 1, 200)
				np.random.shuffle(x) # 打乱200个数据点
				y = 2 * x + np.random.normal(0, 0.05, (200, ))  # 线性分布
				# plot data
				plt.figure(figsize=(12,8))
				plt.scatter(x, y)
				plt.show()
				
				```
		- 数据分布可视化
			- ![](https://img-blog.csdnimg.cn/20190429145840346.png)
	- 划分训练集和验证集
		- 代码
			- ```python
				# 划分数据集
				from sklearn.model_selection import train_test_split
				x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=2019)
				```
	- 模型搭建
		- 代码
			- ```python
				model = Sequential()
				model.add(Dense(units=1, input_dim=1)) 
				model.compile(loss='mse', optimizer='sgd')  # 回归损失函数一般使用mse
				model.summary()
				```
		- 结果
			- 可以看到，这个模型只有一个全连接层作为线性拟合。
			- ![](https://img-blog.csdnimg.cn/20190429150418275.png)
	- 训练及可视化
		- 代码
			- ```python
				# 模型训练
				# 训练方法一
				# for step in range(300):
				#     cost = model.train_on_batch(x_train, y_train)
				#     if step % 100 == 0:
				#         print('train cost: ', cost)
				
				# 训练方法二（一般不使用第一种，因为第二种对batch，loader等配合更好）
				model.fit(x_train, y_train, batch_size=10, verbose=True, epochs=100)
				
				# 验证集上验证训练效果
				cost = model.evaluate(x_valid, y_valid, batch_size=10)
				print('Test cost:', cost)
				W, b = model.layers[0].get_weights()  # 获取训练后的权重
				print('权重参数=', W, '\n偏置=', b)
				
				# 可视化预测结果
				y_pred = model.predict(x_valid)
				plt.scatter(x_valid, y_valid)
				plt.plot(x_valid, y_pred)
				plt.show()
				```
		- 可视化训练结果
			- ![](https://img-blog.csdnimg.cn/20190429151243619.png)
			- 可以看到，100轮训练后有较好的拟合效果，当然数据比较均匀，不然过多训练容易过拟合。
			- 参数只有一个，因为全连接层只设置一个神经元。
- 补充说明
	- 本案例使用Keras框架，这是基于TensorFlow的一个上层框架，如果新手一开始不理解算图编程，Keras是个流行的选择。
	- 本类框架案例均用代码和效果说话，关于神经网络的原理可以见我的其他博客。
	- 具体代码见我的Github，欢迎star或者fork。（开发环境为Jupyter）