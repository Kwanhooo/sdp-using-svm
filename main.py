import os
import pickle

import arff

import numpy as np
from sklearn import svm


def main():
    data_path = "./data/MDP/D''/"
    files = []
    for file in os.listdir(data_path):
        f = os.path.join(data_path, file)
        files.append(f)

    for file in files:
        print('*************************')
        print(file)
        array = np.array(list(arff.load(file)))
        size = array.shape
        print(size)
        train_percent = 0.1
        train_size = int(train_percent * size[0])
        x = array[:train_size, :-1]
        y = array[:train_size, -1]
        print('=== 尝试训练 ===')
        clf = svm.SVC(kernel='linear', verbose=False)
        clf.fit(x, y)
        print('=== 训练完成 ===')
        correct = 0
        total = 0
        print('=== 尝试预测 ===')
        for r in range(train_size, size[0]):
            x = array[r][:-1]
            y = array[r][-1]
            res = clf.predict([x])
            if res[0] == y:
                correct += 1
            total += 1
        ans = (correct / total) * 100
        print('=== 准确率值 ===\n', ans, '%')

        print('=== 导出模型 ===')
        # 导出模型
        # 取最后一个文件名作为模型名
        model_name = './models/' + file.split('/')[-1] + '.model'
        with open(model_name, 'wb') as f:
            pickle.dump(clf, f)
        print('=== 模型位置 ===\n', model_name)
        print('*************************' + '\n\n\n')


if __name__ == '__main__':
    main()
