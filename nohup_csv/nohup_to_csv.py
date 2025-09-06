import re
import csv
from collections import OrderedDict
import os

def parse_training_logs(file_path):
    # 读取文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            log_content = file.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return [], []
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

    # 使用正则表达式找到所有 Training Log 块
    pattern = r'╭─+ Training Log ─+╮\n(.*?)\n╰─+╯'
    blocks = re.findall(pattern, log_content, re.DOTALL)
    
    data_rows = []
    all_keys = OrderedDict()  # 使用 OrderedDict 保持插入顺序
    
    for block in blocks:
        lines = block.strip().split('\n')
        iteration_line = lines[0].strip()
        # 提取 iteration，如 Learning iteration 0/1000000
        iteration_match = re.search(r'Learning iteration (\d+)/(\d+)', iteration_line)
        if not iteration_match:
            continue
        iteration_num = iteration_match.group(1)
        
        row = {'Iteration': iteration_num}
        
        # 跳过空行
        i = 1
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('│') and ':' in line:
                # 去除两端的 │ 和可能的 ANSI 颜色代码
                clean_line = re.sub(r'\x1b\[\d+m', '', line.strip('│ ').strip())
                if ':' in clean_line:
                    key, value = clean_line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    row[key] = value
                    all_keys[key] = None  # 收集所有唯一的键
            i += 1
        
        data_rows.append(row)
    
    return data_rows, list(all_keys.keys())

def save_to_csv(data_rows, keys, output_file='training_logs.csv'):
    if not data_rows:
        print("No data to save.")
        return
    
    # 确保 Iteration 是第一列
    headers = ['Iteration'] + [k for k in keys if k != 'Iteration']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in data_rows:
            writer.writerow(row)
    
    print(f"CSV file saved to: {os.path.abspath(output_file)}")

# 主程序
file_path = "/home/ubuntu/projects/tool_kit/nohup_csv/nohup/nohup_0901_01_GentleWalk"
data_rows, keys = parse_training_logs(file_path)
save_to_csv(data_rows, keys, f"./nohup_csv/csv/{file_path.split('/')[-1]}.csv")