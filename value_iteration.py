from main import *
import time
import gymnasium as gym
import numpy as np

# 假设 env 已经初始化
env = AoIMDPSingleArmEnv(
    lambda_rate_func=lambda_rate_func,  
    ps_func=ps_func,                    
    transmission_delay_func=transmission_delay_func  
)

# 获取状态空间的大小，分别从 Tuple 中提取
aoi_size = env.observation_space.spaces[0].n  # AoI 的空间大小
cache_size = env.observation_space.spaces[1].n  # Cache 的空间大小

# 初始化 Whittle、值函数 V 和策略矩阵
whittle = np.zeros((aoi_size, cache_size))
V = np.zeros((aoi_size, cache_size))  # 初始化状态值函数 V
policy = np.zeros((aoi_size, cache_size), dtype=int)  # 策略初始化为全0

def value_iteration(env, gamma=0.5, theta=1e-3, max_iterations=1000):
    action_space_size = env.action_space.n  # 动作空间大小 (0: 不发送, 1: 尝试发送)
    
    for i in range(max_iterations):
        delta = 0  # 用于记录值函数的变化量
        start_time = time.time()
        for aoi in range(aoi_size):  # 遍历所有可能的 AoI 状态
            for cache in range(cache_size):  # 遍历缓存状态
                state = (aoi, cache)
                v = V[aoi, cache]  # 当前状态的值
                
                # 计算当前状态下的期望值（动作空间）
                Q_values = np.zeros(action_space_size)  # 动作的 Q 值
                for action in range(action_space_size):
                    # 模拟采取这个动作之后的状态和奖励
                    env.state = state  # 强制设置当前状态
                    next_state, reward, _, _ = env.step(action)  # 执行动作获取下一个状态和奖励
                    
                    # 提取下一个状态的值
                    next_aoi, next_cache = next_state
                    Q_values[action] = reward + gamma * V[next_aoi, next_cache] - whittle[aoi, cache] * action
                
                # 更新状态的值函数
                V[aoi, cache] = np.max(Q_values)  # 取最大动作的值
                policy[aoi, cache] = np.argmax(Q_values)  # 记录最优动作
                
                # 记录值函数的最大变化量
                delta = max(delta, np.abs(v - V[aoi, cache]))
        
        iteration_time = time.time() - start_time  
        
        print(f"第 {i+1} 次迭代，变化量 delta: {delta:.6f}，预计剩余时间: {iteration_time * (max_iterations - (i+1)):.2f} 秒")
        if delta < theta:
            print(f"值迭代在第 {i+1} 次迭代后收敛")
            break
    
    return V, policy

def binary_whittle(min_value, max_value, gamma=0.5, theta=1e-3, max_iterations=1000):
    min_whittle = np.full((aoi_size, cache_size), min_value)  
    max_whittle = np.full((aoi_size, cache_size), max_value)  
    
    for iterations in range(max_iterations):
        V, policy = value_iteration(env)
        delta = 0
        start_time = time.time()
        for aoi in range(aoi_size):
            for cache in range(cache_size):
                state = (aoi, cache)
                low, high = min_whittle[aoi, cache], max_whittle[aoi, cache]
                
                # 使用二分法逼近 Whittle Index
                while high - low > theta:
                    mid = (low + high) / 2
                    Q0, Q1 = 0, 0  # 初始化动作的 Q 值
                    
                    for action in [0, 1]:
                        env.state = state  # 强制设置当前状态
                        next_state, reward, _, _ = env.step(action)
                        next_aoi, next_cache = next_state

                        if action == 0:
                            Q0 = reward + gamma * V[next_aoi, next_cache]
                        else:
                            Q1 = reward + gamma * V[next_aoi, next_cache] - mid  # 减去 mid 表示 Whittle Index 的惩罚
                        
                    # 更新二分法的搜索范围
                    if Q1 > Q0:  # 如果动作 1（发送）的价值更大
                        low = mid
                    else:
                        high = mid
                
                # 更新 Whittle Index
                whittle[aoi, cache] = (low + high) / 2
                min_whittle[aoi, cache] =low
                max_whittle[aoi, cache] = high 
                # 记录变化量，便于检测收敛
                delta = max(delta, np.abs(min_whittle[aoi, cache] - max_whittle[aoi, cache]))
        print("######################################################")
        iteration_time = time.time() - start_time  # 计算每次迭代所需时间
        print(f"第 {iterations+1} 次迭代，Whittle Index 变化量 delta: {delta:.6f}，预计剩余时间: {iteration_time * (max_iterations - (iterations+1)):.2f} 秒")
        print("######################################################")
        if delta < theta:
            print(f"Whittle Index 在第 {iterations+1} 次迭代后收敛")
            break
    
    return whittle, V, policy

whittle, V, policy = binary_whittle(-99999, 99999)

save_to_csv(whittle, V, policy, folder_name="value_iteration", prefix="value_iteration")