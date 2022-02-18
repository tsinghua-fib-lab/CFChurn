# -*- coding: utf-8 -*-
"""
@author: zgz

修改dl3中mlflow的结果，从而能传到dl2上使用
"""

import os
import yaml


def getFileNames(path):
    # 列出某个目录下的文件和文件夹，可以是绝对和相对目录
    file_names = []
    for home, dirs, files in os.walk(path):
        for file in files:
            if file == 'meta.yaml':
                file_names.append(os.path.join(home, file))
    return file_names


def configReset(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    data['artifact_uri'] = '/data4/zhangguozhen/churn_prediction/exp_logs/mlflow/7/' + str(data['run_id']) + '/artifacts'
    data['experiment_id'] = '7'

    with open(path, 'w') as f:
        yaml.dump(data, f)


if __name__ == '__main__':
    file_path = r'C:/Users/111/Desktop/disentangle/data/0'
    config_path = getFileNames(file_path)
    for path in config_path:
        configReset(path)
