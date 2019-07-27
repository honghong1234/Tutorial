# 数据增广
@[toc]
## 简介
- 在实际的深度学习项目中，数据集的需求是非常大的，强大的模型包含更多的参数，训练这些参数需要大量的数据；大量的数据训练使得模型的泛化能力变强，一定程度上克制过拟合的出现。
- 数据增广是对数据集进行倍增的有效手段，首次成功应用于AlexNet取得巨大的效果。对图片增广其主要含义为对原来的图片进行翻转（水平翻转和垂直翻转）、扭曲、变形、拉伸、填充、换色、裁减等手段产生新的图片，该图片近似于原图分布且对模型而言是全新的图片，从而达到获取更多训练数据且有标注（标注与原图一致或可以通过变换得到）的目的。
- Keras作为一个成熟的方便的深度学习框架提供了很高效的图片数据增广的API，本项目将逐一演示其作用。
## 原则
- 数据增广的手段是以任务为驱动的，是对任务有利的而不是有害的。
- 不恰当的数据增广会干扰模型的训练，而不会提高模型效果。（如将手写数据垂直翻转没有意义，没有人的手写字是倒置的且会使模型难以拟合。）
- 不是所有时候数据增广都是有效果的，有时候即使正确的增广未必会达到预期的效果，这是模型的问题。
## 数据增广
- 说明
  - Keras要求在创建生成器的时候就指定增广方式，所以对于训练数据和测试数据必须创建不同的生成器（要求训练数据一般情况下是不需要增广的）。**注意，生成器只是生成器，要想生成数据需要调用生成器的flow方法或者flow_\*方法才能得到数据集，调用flow得到的生成器才是fit_generator方法需要的生成器。
  - 参数说明
    - ```python
        from keras.preprocessing.image import ImageDataGenerator
        import keras.backend as K

        train_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            rescale=1/255.,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=0.,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False,
            preprocessing_function=None,
            data_format=K.image_data_format())
        ```
    - 说明
      - featurewise_center：布尔型，输入数据集去中心化（均值为0）。
      - samplewise_center：布尔型，输入数据每个样本均值为0。
      - featurewise_std_normalization：布尔型，将输入样本除以数据集的标准差以完成标准化。
      - samplewise_std_normalization：布尔型，将输入样本除以样本自身的标准差以完成标准化。
      - rescale：数值型，重放缩因子，默认为None。如果为None或0则不进行放缩,否则会将该放缩因子乘到样本数据上(在应用任何其他变换之前），一般设定为1/255.用于图片归一化。
      - zca_whitening：布尔型，对输入数据进行ZCA白化。
      - zca_epsilon: 数值型，ZCA白化使用的eposilon，默认1e-6。
      - rotation_range：整型，增广时图片随机转动的角度，取值为0-180。
      - width_shift_range：数值型，图片宽度的某个比例值，增广时图片水平偏移的幅度。
      - height_shift_range：数值型，图片高度的某个比例值，增广时图片竖直偏移的幅度。
      - shear_range：数值型，剪切强度（逆时针方向的剪切变换角度）。
      - zoom_range：数值型或[low, high]的列表，随机缩放的幅度，数值型表示[low, high]=[1-zoom_range, 1+zoom_range]。
      - channel_shift_range：数值型，通道偏移的幅度。
      - fill_mode：'constant','nearest','reflect'或'wrap'取值之一，当增广越出边界按该值指定的方法处理。
      - cval：数值型，当fill_mode为'constant'时，越界点的填充值。
      - horizontal_flip：布尔型，是否随机进行水平翻转。
      - vertical_flip：布尔型，是否随机进行竖直翻转。
      - preprocessing_function：将被应用于每个输入的函数。该函数将在图片缩放和数据增广之后运行。该函数接受一个参数，为一张图片（ndarray），并且输出一个具有相同shape的ndarray。
      - data_format：'channel_first'或'channel_last'之一，代表图像的通道维的位置。numpy类型图片通道维在最后，如(224,224, 3)。
  - flow方法参数说明
    - ```python
        train_generator = train_gen.flow(
            X,
            y, 
            batch_size=1, 
            shuffle=True, 
            seed=None, 
            save_to_dir=None, 
            save_prefix='', 
            save_format='png')
        ```
    - 说明
      - flow方法会死循环地返回一批随机增广后数据及其标签（在y不为None时）。
      - X：样本数据，四维数据。黑白图像的channel轴的值为1，彩色图像为3。
      - y：与X第一维数值相同的标签数据。
      - batch_size：批尺寸大小，默认为32。
      - shuffle：是否随机打乱数据，默认为True。
      - save_to_dir：默认为None，字符串类型，图片存储目录，该参数能让你将增广后的图片保存本地。
      - save_prefix：字符串类型，保存增广后图片时使用的前缀（如train）, 仅当设置了save_to_dir时生效。
      - save_format：'png'或'jpeg'之一，指定保存图片的数据格式，默认'jpeg'。
      - seed：随机数种子，保证复现。

- 增广
  - 代码
    - ```python
        train_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            rescale=1/255.,
            zca_whitening=False,
            zca_epsilon=1e-6,
            rotation_range=15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=None,
            data_format=K.image_data_format())

        train_generator = train_gen.flow(
            X,
            y, 
            batch_size=1, 
            shuffle=True, 
            seed=None, 
            save_to_dir='gene', 
            save_prefix='train', 
            save_format='png')

        index = 0
        for (batch_x, batch_y) in train_generator:
            index += 1
            if index > 50:
                break
        ```
  - 结果
    - ![](https://img-blog.csdnimg.cn/20190727123052866.png)
    - 可以看到，每一批都有概率进行了随机增广。
## 补充说明
- 本项目重点是数据增广，数据的多种读入方式没有过多提及。
- 具体代码上传至我的Github，欢迎Star或者Fork。
- 如有错误，欢迎指正。