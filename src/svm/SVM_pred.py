import os
import pickle
import numpy as np
import arff


def SVM_pred():
    # 从命令行读取文件
    module = input('请输入要预测的模块路径：')
    metrics_res = "./data/Todo/"
    file = metrics_res + "res.arff"
    metrics = "D:\\Workspace-Java\\SoftwareMetricsAnalyse\\out\\artifacts\\SoftwareMetricsAnalyse_jar\\S.exe " + \
              module + " " + metrics_res
    res = os.system(metrics)
    print(res)

    print('*** ', file, ' ***')
    array = np.array(list(arff.load(file)))
    size = array.shape
    print('        ', size, '\n')

    print('=== 载入模型 ===')
    # 从文件中读取模型
    with open('./models/KC3.arff.model', 'rb') as f:
        clf = pickle.load(f)

    print('=== 模型验证 ===')
    for r in range(0, size[0]):
        x = array[r][:-1]
        print('模块 ' + r.__str__() + '：\n', x)
        res = clf.predict([x])[0]
        print('预测结果：', res)
        print('\n================================================================================\n\n')


