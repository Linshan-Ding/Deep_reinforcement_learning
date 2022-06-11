from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train()  # set training model

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32))  # 转换为tensor
        pdparam = self.forward(x)  # 前向传播
        pd = Categorical(logits=pdparam)  # probability distribution|构建动作概率分布的实例
        action = pd.sample()  # Pi(a|s) in action via pd|依据概率选择动作-动作采样
        log_prob = pd.log_prob(action)  # log_prob of pi(a|s)|动作的对数概率
        self.log_probs.append(log_prob)  # store for traning|事件中动作序列的对数概率列表
        return action.item()

def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0
    # compute the returns efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma*future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets  # 加负号最小化损失
    loss = torch.sum(loss)
    optimizer.zero_grad()  # 梯度清零
    loss.backward()
    optimizer.step()
    return loss

def main():
    env = gym.make('CartPole-v0')
    in_dim = env.observation_space.shape[0]  # 4
    out_dim = env.action_space.n  # 2
    pi = Pi(in_dim, out_dim)  # policy pi_theta for REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr=0.01)
    for epi in range(300):
        state = env.reset()
        for t in range(200):  # cartpole max timestep is 200
            action = pi.act(state)  # 根据当前状态选择动作
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
        # 一个事件路径结束，更新策略参数
        loss = train(pi, optimizer)  # train per episode
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()  # 训练后清除记录的数据
        print(f'Episode {epi}, loss:{loss}, total_reward:{total_reward}, solved:{solved}')

if __name__ == '__main__':
    main()