import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AoIMDPSingleArmEnv(gym.Env):
    """Custom AoI MDP Environment for a single arm problem with dynamic parameters."""
    
    def __init__(self, max_aoi=999, lambda_rate_func=None, ps_func=None, transmission_delay_func=None):
        super(AoIMDPSingleArmEnv, self).__init__()
        
        # 环境参数
        self.max_aoi = max_aoi  # AoI的最大值（防止无界增长）

        # 动态传入的函数
        self.lambda_rate_func = lambda_rate_func  # 动态更新到达率函数
        self.ps_func = ps_func  # 动态传输成功概率函数
        self.transmission_delay_func = transmission_delay_func  # 动态传输延迟函数
        
        # 定义状态空间 (AoI, Cache)
        # AoI 的范围是从 0 到 max_aoi
        # Cache 的状态 0: 缓存为空，1: 缓存有更新包
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.max_aoi + 1),  # AoI 的范围是 0 到 max_aoi
            spaces.Discrete(2)  # 缓存状态只有 0 或 1
        ))
        
        # 动作空间: 0 -> 不发送, 1 -> 尝试发送
        self.action_space = spaces.Discrete(2)
        
        # 初始化状态
        self.state = None
        self.reset()

    def reset(self):
        """重置环境，初始化状态。"""
        self.state = (0, 0)  # 初始 AoI = 0, Cache = 0 (空)
        return self.state, {}

    def step(self, action):
        """
        执行动作，返回新状态、奖励和其他信息。
        
        参数:
        action (int): 0 -> 不发送, 1 -> 尝试发送
        
        返回:
        observation, reward, done, info
        """
        aoi, cache = self.state
        
        # 动态计算当前状态下的参数
        lambda_rate = self.lambda_rate_func(self.state)  # 更新到达率
        ps = self.ps_func(self.state)  # 传输成功概率
        transmission_delay = self.transmission_delay_func(self.state)  # 传输延迟
        
        # 计算新的状态
        if action == 0:  # 不发送，AoI 增加
            aoi = min(aoi + 1, self.max_aoi)  # 限制 AoI 到 max_aoi
            if np.random.rand() < lambda_rate:
                cache = 1  # 有新包到达，缓存中存储最新的更新包
            else:
                cache = cache  # 缓存状态不变
        elif action == 1 and cache == 1:  # 尝试发送
            if np.random.rand() < ps:  # 传输成功
                aoi = transmission_delay  # AoI 重置为传输延迟
                cache = 0  # 传输成功，缓存清空
            else:
                aoi = min(aoi + 1, self.max_aoi)  # 传输失败，AoI 继续增加
                cache = 1  # 缓存保留更新包

        # 计算奖励（负的AoI值，越小越好）
        reward = -aoi
        
        # 检查是否结束（此环境不设终止条件）
        done = False
        
        # 更新状态
        self.state = (aoi, cache)
        
        return self.state, reward, done, {}

    def render(self):
        """渲染当前状态。"""
        aoi, cache = self.state
        print(f"Current AoI: {aoi}, Cache: {cache}")

    def close(self):
        pass
