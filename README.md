# Handwritten-Digit-Recognition
图像识别入门-手写数字识别   
本项目使用mnist数据集进行训练，项目中的0-9文件夹为测试效果时的图片，transformed目录中的0-9文件夹为上述测试图片经过预处理后的图片，你可以使用本人的图片进行测试，也可以使用你自己的图片      
main.py: 为训练及测试，并保存训练模型   
transformed.py: 对图像进行预处理，批量   
recognition.py: 加载模型，读取预处理后的手写数字图片，使用模型识别数字，批量   
recognition_gui.py: 加载模型，读取手写数字图片，使用模型识别数字，图形化（代码中进行预处理，不需要读取预处理后的图片）   
model.pth: 训练好的模型   
本人已经将训练好的模型放在项目中，不需要运行main.py，你也可以删除model.pth和data目录下的所有文件以及目录（仅保留data目录），然后运行main.py进行训练，使用的数据集为mnist数据集，代码将自动下载mnis数据集进行训练   
如果你需要批量识别手写数字图片，在更改transformed.py和recognition.py中的读取图片路径后，先后运行transform.py和recognition.py进行识别   
如果只是单个图片识别，你只需要运行recognition_gui.py，使用图形化界面，不需要额外更改文件内容
> 注意: 测试图片均为白底黑字的图片，为了达到更好的识别效果，需要运行transformed.py进行预处理将图片转化为28x28的黑底白字的图片，然后再运行recognition.py进行识别   
> 此外: 由于mnist数据集为美国学生手写的图片，所以书写习惯与中国的书写习惯有所不同，因此会导致部分数字识别准确率较低，比如6极其容易识别为5
