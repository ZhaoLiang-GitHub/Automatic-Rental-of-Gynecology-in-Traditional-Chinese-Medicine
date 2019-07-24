# Automatic-Rental-of-Gynecology-in-Traditional-Chinese-Medicine
中医妇科自动组方相关
本项目为在某公司实习中参与的中医妇科自动组方项目，具体的项目解释在我的[csdn](https://blog.csdn.net/qq_22235017/article/category/7782712)帖子中，欢迎做相关工作的同学与我邮箱联系1318525510@qq.com

本项目结构为
1. 数据处理
    1. 该文件夹为在本次项目中使用过的数据，具体详情见该文件夹内的readme
2. 实体识别\
    在本次项目中实体识别模块，在本次项目中使用了两种模型，一种是直接调用CRF++模型，一种是BiLstm+CRF模型
    1. CRF\
        直接调用CRF的模型，这部分不是本人参与的，具体内容见该文件夹内的CRF工程文档
    2. BiLSTM+CRF\
        该模块为通过BiLSTM编码器得到语义表示矩阵，将该矩阵通过CEF解码得最终的标注结果，
        
