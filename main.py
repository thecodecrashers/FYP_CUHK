import numpy as np
from aoiarm import AoIMDPSingleArmEnv
import os
import pandas as pd

def save_to_csv(whittle, V, policy, folder_name="results", prefix="results"):
    """
    将 Whittle Index、V、Policy 保存为 CSV 文件，并允许自定义文件夹和文件名前缀。
    
    参数:
    whittle: Whittle Index 矩阵
    V: 值函数矩阵
    policy: 策略矩阵
    folder_name: 保存文件的文件夹名称，默认为 "results"
    prefix: 保存文件的前缀，默认为 "results"
    """
    # 创建存储目录，如果不存在则创建
    directory = os.path.join(os.getcwd(), folder_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 初始化列表来保存每一行的数据
    data = []
    
    # 假设 aoi_size 和 cache_size 都根据 whittle, V 和 policy 的大小推断出来
    aoi_size, cache_size = whittle.shape
    print("HHHHHHHHHHHHHHHHHHH")
    print(aoi_size)
    # 遍历 aoi 和 cache，构建 (aoi, cache) 索引和对应的 whittle, V, policy 值
    for aoi in range(aoi_size):
        for cache in range(cache_size):
            data.append({
                "aoi": aoi,
                "cache": cache,
                "whittle": whittle[aoi, cache],
                "V": V[aoi, cache],
                "policy": policy[aoi, cache]
            })
    
    # 转换为 DataFrame
    df = pd.DataFrame(data)

    # 构建文件的完整路径
    csv_path = os.path.join(directory, f"{prefix}_results.csv")

    # 保存为 CSV 文件
    df.to_csv(csv_path, index=False)

    print(f"保存到CSV文件完成，文件保存在 {csv_path}")

#Possion Distrubution
def lambda_rate_func(state):
    aoi, cache = state
    base_lambda = 0.1 + 0.05 * aoi  
    next_arrival_time = np.random.exponential(1 / base_lambda)
    
    return next_arrival_time

def ps_func(state):
    aoi, cache = state
    base_ps = np.exp(-0.05 * aoi)  
    base_ps = np.clip(base_ps, 0, 1)
    success = np.random.binomial(1, base_ps)  
    
    return success

def transmission_delay_func(state):
    aoi, cache = state
    return 1 if cache == 1 else 2 

