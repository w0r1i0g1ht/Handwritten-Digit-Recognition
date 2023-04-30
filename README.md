# Handwritten-Digit-Recognition
图像识别入门-手写数字识别   
本项目使用mnist数据集进行训练，项目中的0-9文件夹为测试效果时的图片，transformed目录中的0-9文件夹为上述测试图片经过预处理后的图片   
main.py: 为训练及测试，并保存训练模型   
transformed.py: 对图像进行预处理，批量
recognition.py: 加载模型，读取预处理后的手写数字图片，使用模型识别数字，批量   
recognition_gui.py: 加载模型，读取手写数字图片，使用模型识别数字，图形化（代码中进行预处理，不需要读取预处理后的图片）
