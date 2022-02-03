import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import gym
#import argparse
import os
from os.path import exists
import shelve

import time
from TD3_collisionPredictor2 import Actor, Critic, ReplayBuffer, TD3, CollisionPredictor
from collections import deque
import shelve
import gc
import glob
import re
import matplotlib.pyplot as plt



problem = "BipedalWalkerHardcore-v3"

method = "TD3_LG"

folder = problem+'/'+method

start_timestep=1e4

std_noise=0.1

total_episodes = 5000

save_every = 500

sequence_length = 3

encoded_features_number = 20

buffer_capacity = 1000000

collision_col_pred_size = 50

env = gym.make(problem)

# # Set seeds
# seed = 88
# env.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

state = env.reset()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

num_prev_episode = 0

file_buffers = glob.glob(folder+'/*-replay_buf.out')
if len(file_buffers) > 0:
    file_buffers.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    file_buffer = file_buffers[-1]
    match_re = re.search(folder+'/(.+?)-replay_buf.out', file_buffer)
    num_prev_episode = int(match_re.group(1))
    print("num_prev_episode = {}".format(num_prev_episode))

def save(epoch,static,agent,replay_buf):
    filename = str(epoch)
    agent.save(folder,filename)
    replay_buf.save(folder,filename)
    # col_predtor.save(folder,filename)

    my_shelf = shelve.open(folder+"/"+filename+"-static.out",'n')
    try:
        my_shelf['static'] = static
    except TypeError:
        print('ERROR shelving: {}'.format(TypeError))
    my_shelf.close()
    print('Save Static successful') 
    
def load(static,agent,replay_buf):
    if num_prev_episode > 0:
        filename = str(num_prev_episode)
        agent.load(folder,str(num_prev_episode))
        replay_buf.load(folder,str(num_prev_episode))
        col_predtor.load(folder,str(num_prev_episode))

        my_shelf = shelve.open(folder+'/'+str(num_prev_episode)+'-static.out')
        static = my_shelf["static"]
        my_shelf.close()
        print("Load Static file = {}".format(len(static['episodic_rewards'])))
    else:
        print("No Buffer file")
        print("No Weight files")
        print("No Static file")
    return static,agent,replay_buf


# Twin Delayed Deep Deterministic (TD3) policy gradient algorithm
scores_deque = deque(maxlen=100)
scores_array = []
avg_scores_array = []
static ={"ends":[],"episodic_rewards":[]}    

time_start = time.time()                    # Init start time

replay_buf = ReplayBuffer(max_size=buffer_capacity)
agent = TD3(state_dim, action_dim, max_action)
col_predtor = CollisionPredictor(state_dim+action_dim)

static,agent,replay_buf = load(static,agent,replay_buf)
print("len(replay_buf.storage) = {}".format(len(replay_buf.storage)))

timestep_after_last_save = 0
total_timesteps = 0

low = env.action_space.low
high = env.action_space.high

print('Low in action space: ', low, ', High: ', high, ', Action_dim: ', action_dim)

expcount = 0
totrain = 0
collision_pred_other_accs = []
        
for i_episode in range(1, total_episodes+1):
    temp_replay_buffer = []
    timestep = 0
    total_reward = 0
    
    # Reset environment
    init_state = np.zeros(state_dim)
    state = env.reset()
    
    done = False
    after_goal_reward = 0
    info = None
    
    
    while True:
        # env.render()
        # Select action randomly or according to policy
        if total_timesteps < start_timestep:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))
            if std_noise != 0: 
                shift_action = np.random.normal(0, std_noise, size=action_dim)
                action = (action + shift_action).clip(low, high)
        
        # Perform action
        new_state, reward, done, info = env.step(action)
        
        # new_state  =  np.concatenate((state[real_state_dim:state_dim], new_state))
        total_reward += reward                          # full episode reward

        # Store every timestep in replay buffer
        if reward == -100:
            info = 'collision'
            reward = -5
            add_reward = -1
            expcount += 1
            col_predtor.addCollision([np.concatenate((state, action), axis = 0)])
        else:
            add_reward = 0
            reward = 5 * reward


        temp_replay_buffer.append((state, new_state, action, reward, done, info))
            
        # replay_buf.add((state, new_state, action, reward, done, info))
        state = new_state

        timestep += 1     
        total_timesteps += 1
        timestep_after_last_save += 1

        if done:                                       # done ?
            if add_reward == -1 or total_reward < 250:            
                totrain = 1
                for temp in temp_replay_buffer: 
                    replay_buf.add(temp)
            elif expcount > 0 and np.random.rand() > 0.5:
                totrain = 1
                expcount -= 10
                for temp in temp_replay_buffer: 
                    replay_buf.add(temp)
         
            static["ends"].append(info)
            break                                      # save score
    

    scores_deque.append(total_reward)
    static["episodic_rewards"].append(total_reward)

    avg_score = np.mean(scores_deque)
    avg_scores_array.append(avg_score)
    
    # train_by_episode(time_start, i_episode) 
    s = (int)(time.time() - time_start)

    print('Ep. {}, Timestep {},  Ep.Timesteps {}, Score: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02} '\
            .format(i_episode+num_prev_episode, total_timesteps, timestep, \
                    total_reward, avg_score, s//3600, s%3600//60, s%60)) 

    train_collision_flag = False
    collision_weight = 10
    if len(col_predtor.collision_trains) > collision_col_pred_size:
        col_predtor.train(replay_buf, timestep)
        TP,TN,FP,FN,accuracy = col_predtor.evaluate(replay_buf)
        collision_pred_other_accs.append(TN)
        print("Collision TP/TN : FP/FN : accuracy = {}/{} : {}/{} :{}".format(TP,TN,FP,FN,accuracy))

        if len(collision_pred_other_accs) > 20 and np.mean(collision_pred_other_accs[-100:]) > 48:
            train_collision_flag = True
            collision_weight = (1 - np.clip(avg_score/300, 0, 1))*collision_weight
            print("Train agent with CollisionPredictor weight = {}".format(collision_weight))
     
    if totrain == 1:
        agent.train(replay_buf, timestep, collision_predictor=col_predtor, train_collision_flag=train_collision_flag, collision_weight=collision_weight, train_actor=totrain)
    else:
        agent.train(replay_buf, 100, collision_predictor=col_predtor, train_collision_flag=train_collision_flag, collision_weight=collision_weight, train_actor=totrain)

    totrain = 0

    if (i_episode+num_prev_episode) % save_every == 0:
        save(i_episode+num_prev_episode,static,agent,replay_buf)

    # # Save episode if more than save_every=5000 timesteps
    # if timestep_after_last_save >= save_every:

    #     timestep_after_last_save %= save_every            
    #     save(agent, 'checkpnt_seed_88', 'dir_chk')  
    
    # if len(scores_deque) == 100 and np.mean(scores_deque) >= 300.5:
    #     print('Environment solved with Average Score: ',  np.mean(scores_deque) )
    #     break 






print('length of scores: ', len(static["episodic_rewards"]), ', len of avg_scores: ', len(avg_scores))

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores, label="Score")
plt.plot(np.arange(1, len(avg_scores)+1), avg_scores, label="Avg on 100 episodes")
plt.legend(bbox_to_anchor=(1.05, 1)) 
plt.ylabel('Score')
plt.xlabel('Episodes #')
plt.show()

env.close()