import os
import pickle

import arff
import numpy as np

from src.svm.SVM_trainer import standardization


def SVM_pred(module):
    # 从命令行读取文件
    # module = input('请输入要预测的模块路径：')
    metrics_res = "D:\\Workspace-Python\\sdp-using-svm\\data\\Todo\\"
    file = metrics_res + "res.arff"
    metrics = "D:\\Workspace-Java\\SoftwareMetricsAnalyse\\out\\artifacts\\SoftwareMetricsAnalyse_jar\\SMA.exe " + \
              module + " " + metrics_res
    res = os.system(metrics)
    # print(res)

    # print('*** ', file, ' ***')
    array = np.array(list(arff.load(file)))
    size = array.shape
    # print('        ', size, '\n')

    # print('=== 载入模型 ===')
    # 从文件中读取模型
    with open('./models/KC3.arff.rf.model', 'rb') as f:
        clf = pickle.load(f)

    # print('=== 模型验证 ===')
    N = 0
    total = 0
    re = ""
    for r in range(0, size[0]):
        total += 1
        x = array[r][:-1]
        x = standardization(x)
        # print('模块 ' + r.__str__() + '：\n', x)
        res = clf.predict([x])[0]
        # print('预测结果：', res)
        if res == 0:
            re += 'N'
        else:
            re += 'Y'
        if r != size[0] - 1:
            re += ','
        if res == 0:
            N += 1
        # print('\n================================================================================\n\n')
    with open('./result/result.txt', 'w') as f:
        f.write(re.__str__())
    # print('预测出N的总数量：', N)
    # print('预测出Y的总数量：', total - N)
