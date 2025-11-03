import numpy as np
import os
import sys

# 保证输出 UTF-8，避免 Windows PowerShell 打印乱码
sys.stdout.reconfigure(encoding='utf-8')

# 要扫描的目录
directory = "Data"  # 替换成你的文件夹路径
num_preview = 100  # 打印前 100 条

def analyze_file(filepath):
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"文件 {os.path.basename(filepath)} 读取出错: {e}")
        return
    
    dtype = data.dtype
    shape = data.shape
    kind = dtype.kind  # f=float, i=int, u=unsigned int, O=object, U=unicode

    print(f"\n文件: {os.path.basename(filepath)}")
    print(f"  数据类型 (dtype): {dtype}")
    print(f"  形状 (shape): {shape}")

    preview_count = min(num_preview, data.size)

    if kind in {'f', 'i', 'u'}:  # 数值向量
        flattened = data.flatten()
        print("  类型判断: 数值向量")
        print(f"  前 {preview_count} 条数值: {flattened[:preview_count]}")
        print(f"  最大值: {flattened.max()}, 最小值: {flattened.min()}")
    elif kind == 'O' or 'U' in dtype.str:  # 对象或 Unicode 字符串
        print("  类型判断: 文本/对象")
        if data.size > 0:
            print(f"  前 {preview_count} 条内容:")
            for i, item in enumerate(data[:preview_count]):
                print(f"    {i+1}: {str(item)}")
        else:
            print("  文件为空")
    else:
        print("  类型判断: 未知类型")
        if data.size > 0:
            print(f"  前 {preview_count} 条内容:")
            for i, item in enumerate(data[:preview_count]):
                print(f"    {i+1}: {str(item)}")

# 遍历目录
for filename in os.listdir(directory):
    if filename.endswith(".npy"):
        analyze_file(os.path.join(directory, filename))
