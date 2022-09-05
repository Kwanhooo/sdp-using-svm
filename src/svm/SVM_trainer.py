import os
import pickle
import yaml

import arff

import numpy as np
from sklearn import svm

from src.eval.Evaluator import Evaluator


def svm_trainer():
    config = yaml.load(open('./src/svm/config.yml', 'r'), Loader=yaml.FullLoader)
    # configurations
    data_path = config['data_path']
    train_percent = config['train_percent']
    verbose = config['verbose']
    skip_validation = config['skip_validation']

    files = []
    for file in os.listdir(data_path):
        f = os.path.join(data_path, file)
        files.append(f)

    go_on = True

    evaluator = Evaluator()

    for file in files:
        if not go_on:
            break
        print('*** ', file, ' ***')
        array = np.array(list(arff.load(file)))
        size = array.shape
        print('        ', size, '\n')

        train_size = int(train_percent * size[0])
        x = array[:train_size, :-2]
        y = array[:train_size, -1]

        print('=== 尝试训练 ===')
        clf = svm.SVC(kernel='linear', verbose=verbose)
        clf.fit(x, y)

        correct = 0
        total = 0
        if not skip_validation:
            print('=== 模型验证 ===')
            for r in range(train_size, size[0]):
                x = array[r][:-2]
                y = array[r][-1]
                if verbose:
                    print('输入：', x)
                    print('预测：', clf.predict([x])[0])
                    print('实际：', y)
                res = clf.predict([x])
                evaluator.confuse_matrix(res[0], y)
                if res[0] == y:
                    correct += 1
                    if verbose:
                        print('结果：正确', '\n')
                else:
                    if verbose:
                        print('结果：错误', '\n')
                total += 1
            print('=== 评价指标 ===\n')
            evaluator.__str__()
            print('\n===============')

        print('\n=== 导出模型 ===')
        # 导出模型
        # 取最后一个文件名作为模型名
        model_name = './models/' + file.split('/')[-1] + '.model'
        with open(model_name, 'wb') as f:
            pickle.dump(clf, f)
        print(model_name)
        print('\n*********************************' + '\n\n\n')
        go_on = False
        evaluator.reset()
