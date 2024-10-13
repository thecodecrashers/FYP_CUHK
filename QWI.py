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

# 初始化 Q 值矩阵、Whittle 指数 λ 和策略矩阵
Q = np.zeros((aoi_size, cache_size, 2))  # Q 值表，初始化为0
whittle = np.zeros((aoi_size, cache_size))  # 初始化 Whittle 指数 λ
policy = np.zeros((aoi_size, cache_size), dtype=int)  # 策略初始化为全0
V = np.zeros((aoi_size, cache_size))
# 更新参数
alpha_init = 0.1  # Q 值更新的学习率初始值
beta_init = 0.01  # Whittle 指数更新的学习率初始值
gamma = 0.5  # 折扣因子

def qwi_learning(env, episodes=1000000, max_steps=10000, alpha_decay=5000, beta_decay=5000):
    action_space_size = env.action_space.n  # 动作空间大小 (0: 不发送, 1: 尝试发送)
    
    for episode in range(episodes):
        state = env.reset()  # 初始化状态
        (aoi, cache), _ = state  # 获取初始状态
        
        for step in range(max_steps):
            # 计算当前的学习率 alpha 和 beta
            alpha = 1 / (1 + episode // alpha_decay)
            beta = 1 / (1 + (episode // beta_decay) * np.log(episode + 1))

            # epsilon-greedy 策略选择动作
            if np.random.rand() < 0.1:  # 这里选择 0.1 作为探索概率
                action = np.random.choice(action_space_size)  # 随机选择动作
            else:
                action = np.argmax(Q[aoi, cache])  # 根据 Q 值选择最优动作
            
            # 执行动作并获取下一个状态、奖励
            next_state, reward, done, _ = env.step(action)
            next_aoi, next_cache = next_state  # 获取下一个状态

            # 更新 Q 值
            best_next_action = np.argmax(Q[next_aoi, next_cache])  # 下一个状态的最优动作
            td_target = reward + gamma * Q[next_aoi, next_cache, best_next_action]
            td_error = td_target - Q[aoi, cache, action]
            Q[aoi, cache, action] += alpha * td_error  # 更新 Q 值
            
            # 更新 Whittle 指数
            whittle_error = Q[aoi, cache, 1] - Q[aoi, cache, 0]
            whittle[aoi, cache] += beta * whittle_error  # 更新 Whittle 指数
            
            # 更新当前状态
            aoi, cache = next_aoi, next_cache
            
            if done:
                break

        # 每隔一段时间输出一下训练情况
        if (episode + 1) % 100 == 0:
            print(f"第 {episode + 1} 次 episode 完成，当前学习率 alpha: {alpha:.6f}, beta: {beta:.6f}")

    # 根据 Q 值更新状态值函数 V 和策略
    for aoi in range(aoi_size):
        for cache in range(cache_size):
            V[aoi, cache] = np.max(Q[aoi, cache])  # 状态值为最大 Q 值
            policy[aoi, cache] = np.argmax(Q[aoi, cache])  # 策略为最大 Q 值对应的动作
    
    return V, whittle, policy

"""
def qwi_learning(env, episodes=10000, max_steps=100, alpha_decay=5000, beta_decay=5000):
    action_space_size = env.action_space.n  # 动作空间大小 (0: 不发送, 1: 尝试发送)
    
    for episode in range(episodes):
        state = env.reset()  # 初始化状态
        (aoi, cache),_ = state  # 获取初始状态
        for step in range(max_steps):
            # 计算当前的学习率 alpha 和 beta
            alpha = 1 / (1 + episode // alpha_decay)
            beta = 1 / (1 + (episode // beta_decay) * np.log(episode + 1))

            # epsilon-greedy 策略选择动作
            if np.random.rand() < 0.1:  # 这里选择 0.1 作为探索概率
                action = np.random.choice(action_space_size)  # 随机选择动作
            else:
                action = np.argmax(Q[aoi, cache])  # 根据 Q 值选择最优动作
            
            # 执行动作并获取下一个状态、奖励
            next_state, reward, done, _ = env.step(action)
            next_aoi, next_cache = next_state  # 获取下一个状态

            # 更新 Q 值
            best_next_action = np.argmax(Q[next_aoi, next_cache])  # 下一个状态的最优动作
            td_target = reward + gamma * Q[next_aoi, next_cache, best_next_action]
            td_error = td_target - Q[aoi, cache, action]
            Q[aoi, cache, action] += alpha * td_error  # 更新 Q 值
            
            # 更新 Whittle 指数
            whittle_error = Q[aoi, cache, 1] - Q[aoi, cache, 0]
            whittle[aoi, cache] += beta * whittle_error  # 更新 Whittle 指数
            
            # 更新当前状态
            aoi, cache = next_aoi, next_cache
            
            if done:
                break

        # 每隔一段时间输出一下训练情况
        if (episode + 1) % 100 == 0:
            print(f"第 {episode + 1} 次 episode 完成，当前学习率 alpha: {alpha:.6f}, beta: {beta:.6f}")

    # 根据 Q 值更新策略
    for aoi in range(aoi_size):
        for cache in range(cache_size):
            policy[aoi, cache] = np.argmax(Q[aoi, cache])  # 策略为最大 Q 值对应的动作
    
    return Q, whittle, policy
    """

# 使用 Q-learning with Whittle Index 更新 Q、Whittle 指数和策略
V, whittle, policy = qwi_learning(env)
print(whittle.shape)
# 保存结果
save_to_csv( whittle,V, policy, folder_name="qwi_learning", prefix="qwi_learning")
