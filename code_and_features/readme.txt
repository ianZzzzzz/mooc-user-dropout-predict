Data_process_Pipline.py ：
	特征工程代码，包含数据清洗与特征提取模块，
	需要调用raw_data_file文件夹内的数据
	内存占用8G左右，耗时一小时左右。

model.py ：
	模型预测与展示代码，调用特征工程提取后的特征作为训练数据。
	模型训练调用CPU总耗时一分钟左右

after_processed_data_file文件夹：
	包含特征工程的中间文件与最终提取的特征数据集
	feature：特征数据集 由model.py调用

raw_data_file：
	原始数据文件夹，来源：http://moocdata.cn/data/user-activity
	请将此文件夹与.py文件放置于同一路径下
	此文件夹中存在文件名以 _T.csv 结尾的文件，为小规模测试文件，从代码中更改路径即可调用

为了方便复现预测结果，已先将提取的特征数据集存储于feature文件夹中，
可从model.py 中直接读取，用于训练模型。