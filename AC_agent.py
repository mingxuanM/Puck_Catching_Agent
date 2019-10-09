#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AC_agent.py
RL training process
AC_agent: RL agent class, contains all networks and training function.
    get_q_value()
    train_network()
    get_action()
    set_epsilon()

Interaction_env: the environment class that the agent interacts with.
    reset()
    act()
    reward_cal()
    action_generation()
"""
import argparse
from interaction_env import Interaction_env
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import time
import sys
import json

from config import n_actions, RQN_num_feats, action_length, qlearning_gamma
# device = "cuda:0"
# n_actions = 6 # 1 no action + 4 directions acc + 1 click
# qlearning_gamma = 0.9
# # n_actions = 4*2 # 4 directions * 2 if click
# action_length = 5 # frames
# RQN_num_feats = 22 # 4 caught object + 2 mouse + 4*4

# Workflow:
# learning_agent.get_action(state_t) -> action -> 
# env.step(action) -> (state_next, reward, is_done) ->
# target_agent.get_target(state_next, reward) -> target ->
# learning_agent.train_step(state_t, action, target) -> loss




class _RQN(nn.Module):
    def __init__(self, in_dim=RQN_num_feats, out_dim=n_actions):
        super(_RQN, self).__init__()
        self.hidden_dim = 30
        self.batch_size = 1
        self.num_layers = 1

        self.lstm0 = nn.LSTM(in_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.relu0 = nn.ReLU()
        self.dense1 = nn.Linear(self.hidden_dim, out_dim)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        x,_ = self.lstm0(x.float())
        x = self.relu0(x[:,-1])
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        return x

class _Policy(nn.Module):
    def __init__(self, in_dim=RQN_num_feats, out_dim=n_actions):
        super(_Policy, self).__init__()
        self.hidden_dim = 30
        self.batch_size = 1
        self.num_layers = 1

        self.lstm0 = nn.LSTM(in_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.relu0 = nn.ReLU()
        self.dense1 = nn.Linear(self.hidden_dim, out_dim)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(out_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x,_ = self.lstm0(x.float())
        x = self.relu0(x[:,-1])
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

class AC_agent():
    def __init__(self, gamma, input_frames=action_length, in_dim=RQN_num_feats, out_dim=n_actions, lr=5e-4, device=torch.device("cpu")):
        self.device = device
        self.cpu = torch.device("cpu")

        self.critic = _RQN(in_dim, out_dim).to(self.device)
        self.critic = self.critic.float()
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.criterion_critic = nn.MSELoss()

        self.actor = _Policy(in_dim, out_dim).to(self.device)
        self.actor = self.actor.float()
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.criterion_actor = self._max_loss()


        self.gamma = torch.tensor(gamma).float().to(self.device)
        # self.epsilon = 1
    
    def _max_loss(self, policy, action, q_value):
        # action = torch.argmax(policy)
        return -torch.log(policy[:,action]) * q_value

    # return q value for a given state & action pair, all in device
    def get_q_value(self, state_t, action):
        # q_values = self.critic.forward(torch.from_numpy(state_t).to(self.device))
        q_values = self.critic.forward(state_t)
        return q_values[:,action]

    # train network
    # update actor with -log(A(state_t))*Q(state_t, action) first
    # then update critic with r + gamma*Q(state_next, action_next)
    # All in device
    def train_network(self, state_t, action, reward, state_next):
        state_t = torch.from_numpy(state_t).to(self.device)
        policy = self.actor.forward(state_t)
        q_value = self.get_q_value(state_t, action)
        loss_actor = self.criterion_actor(policy, q_value)
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        state_next = torch.from_numpy(state_next).to(self.device)
        action_next = self.get_action(self, state_next)
        target = torch.tensor(reward).float().to(self.device) + self.gamma * self.get_q_value(state_next, action_next)
        loss_critic = self.criterion_critic(q_value, target)
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        return action_next

    # sample action for a given state
    def get_action(self, state_t):
        # thre = np.random.rand()
        # if thre < self.epsilon:
        #     action = np.random.choice(n_actions, 1)[0]
        # else:
        policy = self.actor.forward(torch.from_numpy(state_t).to(self.device))
        action = torch.argmax(policy).to(self.cpu).item()
        return int(action)
    
    # def set_epsilon(self, epsilon_):
    #     self.epsilon = epsilon_

# run one episode
# t_max: maximum running time
# train：if True, calculate loss and call train_step
def train_iteration(learning_agent, target_agent, env, t_max, train=False):
    
    session_reward = []
    td_loss = []
    s = env.reset() # first 10 frames * 22 num_feats
    s = s.reshape((1,action_length,RQN_num_feats))
    t = 0
    is_done = False
    a = learning_agent.get_action(s)
    while t < t_max and not is_done:
        # a = learning_agent.get_action(s)
        # print('action')
        # print(a)
        s_next, reward, is_done = env.act(a)
        # s_next = trajectory # 5 frames * 22 num_feats
        s_next = s_next.reshape((1,action_length,RQN_num_feats))
        reward = np.array([reward])

        if train:
            a_next = learning_agent.train_network(s, a, reward, s_next):
        else:
            a_next = learning_agent.get_action(s_next)
            
        session_reward.append(reward)
        s = s_next
        a = a_next
        t += action_length
    trajectory = env.destory()
    
    return session_reward, td_loss, is_done, trajectory

# Top level training loop, over epochs
def train_loop(learning_agent, target_agent, env, episode, train, timeout, continue_from=0, save_model=False):
    succeed_episode = 0
    time_taken = []
    trajectory_history = []
    for i in range(episode):
        # print('[session {} started] '.format(i) + time.strftime("%H:%M:%S", time.localtime()))
        session_reward, is_done, trajectory = train_iteration(learning_agent, target_agent, env, timeout, train)
        if not train:
            trajectory_history.append(trajectory)
        session_reward_mean = np.mean(session_reward)
        print('[session {} finished] '.format(i+1) + time.strftime("%H:%M:%S", time.localtime()) + ';\t actions taken = {:.4f};\t mean reward = {:.4f};\t total reward = {:.4f};\t epsilon = {:.4f}'.format(
            len(session_reward), session_reward_mean, np.sum(session_reward), learning_agent.epsilon))
        
        if train:
            if i%2==0:
                target_agent.rqn.load_state_dict(learning_agent.rqn.state_dict())
                # learning_agent.set_epsilon(max(learning_agent.epsilon * epsilon_decay, 0.01))
                learning_agent.set_epsilon(max(1-i/episode, 0.01))
            if i%100==0 and i>0 and save_model:
                save_path = './exported/rqn_{}_epoch'.format(i + 1 + args.continue_from)
                torch.save(learning_agent.rqn.state_dict(), save_path)
                print('Model saved in path: ' + save_path)
        # Count and print for catching records
        if is_done:
            succeed_episode += 1
            time_taken.append(len(session_reward))
    print('Agent succeed in catching object in {}/{} ({:.4f}%) episodes'.format(succeed_episode, episode, succeed_episode/episode*100))
    print('End of training, average actions to catch: {}'.format(np.mean(time_taken)))

    if not train:
        with open('./episode_records/trajectory_history.json', 'w') as data_file:
            json.dump(trajectory_history, data_file, indent=4)
        return trajectory_history

    if save_model and train:
        save_path = './exported/rqn_{}_epoch'.format(episode + args.continue_from)
        torch.save(learning_agent.rqn.state_dict(), save_path)
        print('Model saved in path: ' + save_path)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='training a recurrent q-network')
    parser.add_argument('--episode', type=int, action='store',
                        help='number of epoches to train', default=50)
    
    parser.add_argument('--save_model', type=bool, action='store', 
                        help='save trained model or not', default=False)

    parser.add_argument('--train', type=bool, action='store',
                        help='if to train a model', default=False)
    
    # parser.add_argument('--epsilon', type=float, action='store',
    #                     help='epsilon for Q learning', default=0.99)

    parser.add_argument('--lr', type=float, action='store',
                        help='learning rate for Adam optimiser', default=5e-4)

    parser.add_argument('--timeout', type=int, action='store',
                        help='max number of frames for one episode, 1/60s per frame', default=1800)

    parser.add_argument('--continue_from', type=int, action='store',
                        help='continue training from previous trained model', default=0)

    args = parser.parse_args()


    # RQN_num_feats = 22
    # input_frames = 5

    environment = Interaction_env()

    is_cuda = torch.cuda.is_available()
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    # device = torch.device("cpu")

    # initialize learning_agent
    # train for catching pucks
    if args.train:
    
        learning_agent = AC_agent(qlearning_gamma, action_length, RQN_num_feats, n_actions, args.lr, device)
        # environment.predictor.saver.restore(sess, "./model_predictor/checkpoints/pretrained_model_predictor_2.ckpt")
        if args.continue_from > 0:
            checkpoint = torch.load('./exported/ac_{}_epoch'.format(args.continue_from))
            learning_agent.actor.load_state_dict(checkpoint)
            learning_agent.critic.load_state_dict(checkpoint)
    else:
        if args.continue_from == 0:
            sys.exit('[ERROR] test model not specified')
        
        learning_agent = AC_agent(qlearning_gamma, action_length, RQN_num_feats, n_actions, args.lr, device)
        checkpoint = torch.load('./exported/ac_{}_epoch'.format(args.continue_from))
        learning_agent.actor.load_state_dict(checkpoint)
        learning_agent.critic.load_state_dict(checkpoint)

    # train
    _ = train_loop(learning_agent, environment, args.episode, args.train, args.timeout, args.continue_from, args.save_model)