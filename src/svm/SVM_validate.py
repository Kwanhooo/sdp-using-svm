import pickle

import numpy as np
import arff


def SVM_validate():
    with open('./models/KC3.arff.model', 'rb') as f:
        clf = pickle.load(f)

    file = "./data/MDP/D-smote/training/KC3.arff"
    array = np.array(list(arff.load(file)))
    size = array.shape
    total = 0
    correct = 0
    N = 0
    for r in range(0, size[0]):
        x = array[r][:-2]
        y = array[r][-1]
        print('输入：', x)
        res = clf.predict([x])[0]
        print('预测：', res)
        print('实际：', y)
        if res == 'N':
            N+= 1
        if res == y:
            print('结果：正确', '\n')
            correct += 1
        else:
            print('结果：错误', '\n')
        total += 1
    print('\n\n正确率：', correct / total)
    print('N的数量：', N)
    print('Y的数量：', total - N)
