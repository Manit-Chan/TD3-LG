import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import shelve
import random

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

# Code based on: 
# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# https://github.com/sfujim/TD3/blob/master/TD3.py
# Paper: https://arxiv.org/abs/1802.09477

def save_ckp(ckp, name, checkpoint_dir):
    f_path = checkpoint_dir+'/'+str(ckp['epoch'])+'-'+name+'_ckp.pth'
    torch.save(ckp, f_path)

def load_ckp(ckp_fpath, model, optimizer):
    ckp = torch.load(ckp_fpath)
    model.load_state_dict(ckp['state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckp['optimizer'])
        return model, optimizer, ckp['epoch']
    else:
        return model, ckp['epoch']    

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Actor Neural Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x
    
# Q1-Q2-Critic Neural Network    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2



    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1 
    


# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind: 
            X, Y, U, R, D, I = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1) 

    def save(self, folder, filename):
        my_shelf = shelve.open(folder+'/'+filename+"-replay_buf.out",'n')
        try:
            my_shelf['replay_buf'] = self
            print('Save Buffer successful')
        except TypeError:
            print('ERROR shelving: {}'.format(TypeError))
        my_shelf.close()
    
    def load(self, folder, filename):
        my_shelf = shelve.open(folder+'/'+filename+'-replay_buf.out')
        replay_buf = my_shelf['replay_buf']
        self.storage = replay_buf.storage
        self.max_size = replay_buf.max_size
        self.ptr = replay_buf.ptr

        my_shelf.close()
        print("Load replay_buf from ep {} successfully".format(filename))  
    
class TD3(object): 
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, iterations, collision_predictor, batch_size=100, discount=0.99, \
              tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2, train_collision_flag=False, collision_weight=1, train_actor =1):
        
        Q_collisions_after = []
        Q_collisions = []
        Q_collisions_after = []
        for it in range(iterations):

            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Delayed policy updates
            if it % policy_freq == 0 and train_actor ==1:

                # Compute actor loss
                # actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                actions = self.actor(state)
                Q = self.critic.Q1(state, actions)
               
                if train_collision_flag:
                    predict_labels = collision_predictor.predict(torch.cat([state, actions], dim=1).detach().cpu().numpy())
                    predict_labels = torch.FloatTensor(predict_labels).to(device)
                    punish_matrix = predict_labels*collision_weight
                    punish_matrix = torch.reshape(punish_matrix,(len(punish_matrix),1))
                    if torch.count_nonzero(predict_labels) > 0:
                        non_zero_idxs = torch.nonzero(predict_labels).split(1, dim=1)
                        Q_collisions.append(torch.mean(Q[non_zero_idxs]))
                        Q_collisions_after.append(torch.mean((Q-punish_matrix)[non_zero_idxs]))
                       
                    actor_loss = -(Q-punish_matrix).mean()
                else:
                    actor_loss = -Q.mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        if train_collision_flag and len(Q_collisions) and len(Q_collisions_after):
            print("Mean Q_collisions = {} / Mean after weight = {}".format(torch.mean(torch.stack(Q_collisions)),torch.mean(torch.stack(Q_collisions_after))))

    def save(self, folder, filename):
        torch.save(self.actor.state_dict(), '%s/%s-actor.pth' % (folder, filename))
        torch.save(self.critic.state_dict(), '%s/%s-critic.pth' % (folder, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s-actor_t.pth' % (folder, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s-critic_t.pth' % (folder, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s-actor_optimizer.pth' % (folder, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s-critic_optimizer.pth' % (folder, filename))
        print("Save Actor Critic Weights successfully")

    def load(self, folder, filename):
        self.actor.load_state_dict(torch.load('%s/%s-actor.pth' % (folder, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s-actor_optimizer.pth' % (folder, filename)))
        self.actor.train()
        self.critic.load_state_dict(torch.load('%s/%s-critic.pth' % (folder, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s-critic_optimizer.pth' % (folder, filename)))
        self.critic.train()
        self.actor_target.load_state_dict(torch.load('%s/%s-actor_t.pth' % (folder, filename)))
        self.actor_target.train()
        self.critic_target.load_state_dict(torch.load('%s/%s-critic_t.pth' % (folder, filename)))
        self.critic_target.train()
        print("Load Actor Critic Weights from ep {} successfully".format(filename))  


class CollisionPredictor(nn.Module):
    def __init__(self,state_act_dim):
        super(CollisionPredictor, self).__init__()
        self.predictor = SGDClassifier(loss='log')
        self.scaler = StandardScaler()
        self.state_act_dim = state_act_dim
        self.collision_trains = np.empty((0,self.state_act_dim), np.float64)

    def predict(self, state_actions):
        batch = self.scaler.transform(state_actions)
        predictions = self.predictor.predict(batch)
        return predictions

    def evaluate(self, replay_buf, batch_size=100):
        not_collision_trains = np.empty((0,self.state_act_dim), np.float64)
        while True:
            chk = False
            ind = np.random.randint(0, len(replay_buf.storage), size=1000)
            for i in ind:
                state, new_state, action, reward, done, info = replay_buf.storage[i]
                if len(not_collision_trains) < int(batch_size/2):
                    if reward != -5 and info != 'collision':
                        not_collision_trains = np.append(not_collision_trains, [np.concatenate((state, action), axis = 0)], axis=0)
                else:
                    chk = True
                    break
            if chk == True:
                break

        idxs = np.random.randint(0, len(self.collision_trains), size=int(batch_size/2))
        batch_collision_trains = self.collision_trains[idxs]
        batch = np.concatenate((not_collision_trains, batch_collision_trains), axis = 0)
        labels = np.concatenate((np.zeros((len(not_collision_trains), 1)), np.ones((len(batch_collision_trains), 1))), axis = 0).ravel()
        batch,labels = unison_shuffled_copies(batch,labels)
        predictions = self.predict(batch)

        TN, FP, FN, TP = confusion_matrix(labels, predictions).ravel()
        
        accuracy =  (TP+TN) /(TP+FP+TN+FN)
        # print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
        return TP,TN,FP,FN,accuracy

    def addCollision(self,state_action):
        self.collision_trains = np.append(self.collision_trains, state_action, axis=0)

    def train(self,replay_buf, iterations, batch_size=100): 
        for it in range(iterations): 
            not_collision_trains = np.empty((0,self.state_act_dim), np.float64)
            while True:
                chk = False
                ind = np.random.randint(0, len(replay_buf.storage), size=1000)
                for i in ind:
                    state, new_state, action, reward, done, info = replay_buf.storage[i]
                    if len(not_collision_trains) < int(batch_size/2) :
                        if reward != -5 and info != 'collision':
                            not_collision_trains = np.append(not_collision_trains, [np.concatenate((state, action), axis = 0)], axis=0)
                    else:
                        chk = True
                        break
                if chk == True:
                    break

            idxs = np.random.randint(0, len(self.collision_trains), size=int(batch_size/2))
            batch_collision_trains = self.collision_trains[idxs]
            batch = np.concatenate((not_collision_trains, batch_collision_trains), axis = 0)
            labels = np.concatenate((np.zeros((len(not_collision_trains), 1)), np.ones((len(batch_collision_trains), 1))), axis = 0).ravel()
            batch,labels = unison_shuffled_copies(batch,labels)
            self.scaler.partial_fit(batch)
            batch = self.scaler.transform(batch)
            self.predictor.partial_fit(batch, labels, classes=[0,1])
