import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import matplotlib.pyplot as plt
import pandas as pd
import csv
import re
import numpy as np
def extract_number_before_csv(filename):
    match = re.search(r'(\d+)(?=\.csv)', filename)
    if match:
        return match.group(1)
    else:
        return None
def find_first_number_below_threshold(directory, threshold=0.5):
    results = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    csv_reader = csv.reader(csvfile)

                    # 跳过表头
                    next(csv_reader, None)

                    # 获取第一列的序号和第二列的浮点数
                    data = [(int(row[0]), float(row[1])) for row in csv_reader]

                    # 找到第一个小于阈值的序号
                    first_below_threshold_number = next((row[0] for row in data if row[1] < threshold), None)

                    # 将结果存储在字典中
                    results[file] = first_below_threshold_number

    return results

def find_significant_outliers(result_dict, threshold_factor=2.5):
    # 提取所有序号
    all_numbers = [number for number in result_dict.values() if number is not None]

    # 计算均值和标准差
    mean_value = np.mean(all_numbers)
    std_dev = np.std(all_numbers)

    # 设置离群值阈值
    threshold = mean_value + threshold_factor * std_dev
    print(mean_value)
    print(std_dev)

    # 找到显著大于阈值的离群值的序号
    significant_outliers = {file: number for file, number in result_dict.items() if number is not None and number > threshold}
    if not significant_outliers:
        return "clean"
    return significant_outliers

def get_all_subdirectories(folder):
    subdirectories = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    return subdirectories


def extract_last_number_from_folder_path(folder_name):
    # 检查文件夹名是否包含 "clean" 字样
    if "clean" in folder_name.lower():
        return "clean"
    # 使用正则表达式提取文件夹名中的数字
    match = re.search(r'\d+$', folder_name)
    if match:
        return match.group()
    # 如果没有匹配到 "clean" 字样且没有数字，则返回 None
    return None
# 用法示例：
folder = f'/home/zq/projects/FreeEagle/record/mnist-SimpleCNN/csv'
subdirectories = get_all_subdirectories(folder)
correct = 0
for  folder_path in subdirectories:
    print('for model:',extract_last_number_from_folder_path(folder_path))
    threshold_value = 0.5
    result_dict = find_first_number_below_threshold(folder_path, threshold_value)
    outliers = find_significant_outliers(result_dict)
    judge = ""
    if outliers == "clean":
        judge = "clean"
        print("clean")
    elif len(outliers) == 1:
        for file, number in outliers.items():
            print(file,number)
            judge = str(extract_number_before_csv(file))
    if judge == str(extract_last_number_from_folder_path(folder_path)):
        correct = correct + 1
    else:
        print(folder_path)
    print("-----------------------------------------")
print("sum:", len(subdirectories),"correct:", correct)


