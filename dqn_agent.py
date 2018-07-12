#first just do DQN and get working with experience replay

import random
import math

import numpy as np
import pandas as pd
import os

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import copy


_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_HOSTILE = 4
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_MARINE = 48
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
INCREASE_ARMY_SUPPLY_REWARD = 0.1
INCREASE_BUILDING_NUM_REWARD = 0.1
GAME_SCORE_REWARD = 0.1

DECREASE_SCV_SUPPLY_PEN = - 0.1

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,#0
    ACTION_SELECT_SCV,#1
    ACTION_BUILD_SUPPLY_DEPOT,#2
    ACTION_BUILD_BARRACKS,#3
    ACTION_SELECT_BARRACKS,#4
    ACTION_BUILD_MARINE,#5
    ACTION_SELECT_ARMY,#6
    ACTION_ATTACK#7
]

_NOT_QUEUED = [0]
_QUEUED = [1]

class DqnAgent(base_agent.BaseAgent):
    def __init__(self):
        super(DqnAgent, self).__init__()
        
        self.qlearn = DQN(actions=list(range(len(smart_actions))))#QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_army_supply = 0
        self.previous_scv_supply = 0
        self.previous_game_score = 0
        
        self.previous_action = None
        self.previous_state = None
        
        self.nextState_scv_started_moving = False
        self.nextState_marine_started_training = False
        self.nextState_marine_started_moving = False
        
        self.gameScoreRecord = []
        self.winLossRecord = []
        self.stepCounter = 0
        
    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]
        
        return [x + x_distance, y + y_distance]
        
    def reset(self):
        super(DqnAgent, self).reset()
        self.stepCounter = 0
        print("END oF EPISODE", self.previous_game_score)
        self.gameScoreRecord.append(self.previous_game_score)
        with open('scores.txt', 'w') as file:
            for item in self.gameScoreRecord:
                file.write("%s\n" % item)
            
        
    def step(self, obs):
        self.stepCounter += 1 
        super(DqnAgent, self).step(obs)
        player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0
        
        unit_type = obs.observation['screen'][_UNIT_TYPE]

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = 1 if depot_y.any() else 0

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = 1 if barracks_y.any() else 0
            
        supply_limit = obs.observation['player'][4]
        army_supply = obs.observation['player'][5]
        scv_supply = obs.observation['player'][6]
        unit_selected = obs.observation['single_select'][0][0] 
        
        scv_started_moving = self.nextState_scv_started_moving
        marine_started_moving = self.nextState_marine_started_moving
        marine_started_training = self.nextState_marine_started_training
        
        killed_unit_score = obs.observation['score_cumulative'][5]
        killed_building_score = obs.observation['score_cumulative'][6]
        building_score = obs.observation['score_cumulative'][4]
        spent_mineral_score = obs.observation['score_cumulative'][11]
        game_score = killed_unit_score + killed_building_score + spent_mineral_score#obs.observation['score_cumulative'][0]
        if self.stepCounter % 200 == 0:
            print("SCORE = ",game_score)
            print("killed_unit_score = ",killed_unit_score)
            print("killed_building_score = ",killed_building_score)
            print("spent_mineral_score = ",spent_mineral_score)
        current_state = [
            supply_depot_count,
            barracks_count,
            supply_limit,
            army_supply,
            unit_selected,
            scv_started_moving,
            marine_started_moving,
            marine_started_training
        ]
        
        if self.previous_action is not None:
            reward = 0
                
            if game_score > self.previous_game_score:
                reward += GAME_SCORE_REWARD
            '''
            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD
            
            
                    
            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD
                
            if army_supply > self.previous_army_supply:
                reward += INCREASE_ARMY_SUPPLY_REWARD
            if scv_supply < self.previous_scv_supply:
                reward += DECREASE_SCV_SUPPLY_PEN
            if building_score > self.previous_building_score:
                #print(INCREASE_BUILDING_NUM_REWARD)
                reward += INCREASE_BUILDING_NUM_REWARD
                #print(reward,"-------------------------------------------------------------------------------------444")
            '''
            #print("reward",reward)
            self.qlearn.learn(self.previous_state, self.previous_action, reward, current_state)
        
        
        rl_action = self.qlearn.choose_action(current_state,obs.observation['available_actions'])
        smart_action = smart_actions[rl_action]
        #print(smart_action)
        
        self.previous_killed_unit_score = killed_unit_score
        self.previous_game_score = game_score
        self.previous_killed_building_score = killed_building_score
        self.previous_army_supply = army_supply
        self.previous_scv_supply = scv_supply
        
        self.previous_building_score = building_score
        self.previous_state = current_state
        self.previous_action = rl_action
        
        self.nextState_scv_started_moving = False
        self.nextState_marine_started_training = False
        self.nextState_marine_started_moving = False
        
        if smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        elif smart_action == ACTION_SELECT_SCV:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_SUPPLY_DEPOT:
            if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)
                    self.nextState_scv_started_moving = True
                    return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_BARRACKS:
            if _BUILD_BARRACKS in obs.observation['available_actions']:
                unit_type = obs.observation['screen'][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
                
                if unit_y.any():
                    target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)
                    self.nextState_scv_started_moving = True
                    return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
    
        elif smart_action == ACTION_SELECT_BARRACKS:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_BARRACKS).nonzero()
                
            if unit_y.any():
                target = [int(unit_x.mean()), int(unit_y.mean())]
        
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        
        elif smart_action == ACTION_BUILD_MARINE:
            if _TRAIN_MARINE in obs.observation['available_actions']:
                self.nextState_marine_started_training = True
                return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        
        elif smart_action == ACTION_SELECT_ARMY:
            if _SELECT_ARMY in obs.observation['available_actions']:
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
        
        elif smart_action == ACTION_ATTACK:
            if _ATTACK_MINIMAP in obs.observation["available_actions"]:
                if self.base_top_left:
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])
                if unit_selected == _TERRAN_SCV:
                    self.nextState_scv_started_moving = True
                elif unit_selected == _TERRAN_MARINE:
                    self.nextState_marine_started_moving = True
                return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])
            
        return actions.FunctionCall(_NO_OP, [])

class DQN: 
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.8): #l4 was 0.01, reward decay 0.9, e greedy 0.8
        self.actions = actions
        #self.lr = learning_rate
        #self.gamma = reward_decay
        
        self.epsilon = e_greedy
        #self.counter = 0
        #self.actionHistory = []
        
        self.batch_size = 32#6#32 #How many experiences to use for each training step.
        self.update_freq = 4 #How often to perform a training step.
        
        self.update_freq_modelNet = 40
        
        self.y = .99 #Discount factor on the target Q-values
        startE = 1 #Starting chance of random action
        self.endE = 0.1 #Final chance of random action
        annealing_steps = 50000. #How many steps of training to reduce startE to endE.
        num_episodes = 10000 #How many episodes of game environment to train network with.
        self.pre_train_steps = 34 #How many steps of random actions before training begins.
        max_epLength = 50 #The max allowed length of our episode.
        load_model = False #Whether to load a saved model.
        path = "./dqn" #The path to save our model to.
        h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
        tau = 0.001 #Rate to update target network toward primary network

        tf.reset_default_graph()
        self.mainQN = Qnetwork(h_size)
        self.targetQN = Qnetwork(h_size)
        
        
        self.exploreRewardConst = 0.01

        

        saver = tf.train.Saver()

        trainables = tf.trainable_variables()

        self.targetOps = updateTargetGraph(trainables,tau)
        print("before")
        self.modelNet = modelNetwork()
        print("after")
        myBuffer = experience_buffer()
        init = tf.global_variables_initializer()
        #Set the rate of random action decrease. 
        self.e = startE
        self.stepDrop = (startE - self.endE)/annealing_steps

        #create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        self.total_steps = 0

        #Make a path for our model to be saved in.
        if not os.path.exists(path):
            os.makedirs(path)
            
        self.sess = tf.Session()
        self.sess.run(init)
        self.episodeBuffer = experience_buffer()

    def choose_action(self, state, available_actions):
        #print("available_actions",available_actions)
        #time.sleep(1.0)
        available_smart_actions = getAvailableSmartActions(available_actions)
        
        if np.random.uniform() < self.e:#self.epsilon:
            # choose best action
            #state_action = #self.q_table.ix[observation, :]
            ##state = state.astype(int)
            action = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:[state]})[0]
            
            
            if self.total_steps % (200) == 0 and len(self.episodeBuffer.buffer) > 0:
            
                A = self.episodeBuffer.buffer
                #print("afirst",A)
                A = np.asarray(A)[:,0]
                #print("asec",A)
                #B = np.array(list(set([tuple(x) for x in A])))
                #B = A[np.unique(A.view("int, int, int, int"), return_index=True)[1]]
                B = [A[0]]
                for data in A:
                    matches = False
                    #print(A)
                    #print(B)
                    for i in B:
                        #print("hre",np.array_equal([[0,3,True,False],[0,3,True,False]]))
                        #print("ok",([0,3,True,False]==[0,3,True,False]))
                        #print("datas",data,i,np.all([data,i]))
                        if ((data==i)):
                            matches = True
                    if not matches:
                        #print("no match", matches, data)
                        B.append(data)
                #tuple(B)
                for i in B:
                    value = self.sess.run(self.mainQN.Value,feed_dict={self.mainQN.scalarInput:[i]})[0]
                    actionPrint = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:[i]})[0]
                    print(i,value,actionPrint)
                    
                #time.sleep(2.0)
                #print()
            # some actions have the same value
            #state_action = #state_action.reindex(np.random.permutation(state_action.index))
            
            #action = #state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(available_smart_actions)#self.actions)
        
        return action

    def learn(self, previous_state, previous_action, reward, current_state):
        
        #lastActionDidSomething = previous_state != current_state
        self.episodeBuffer.add(np.reshape(np.array([previous_state,previous_action,reward,current_state]),[1,4]))
        self.total_steps += 1
        #print(self.total_steps, self.e)
        #print(self.episodeBuffer.buffer)
        if self.total_steps > self.pre_train_steps:
            if self.e > self.endE:
                self.e -= self.stepDrop
            
            if self.total_steps % (self.update_freq) == 0:
                trainBatch = self.episodeBuffer.sample(self.batch_size) #Get a random batch of experiences.
                
                #Below we perform the Double-DQN update to the target Q-values
                #print("np.vstack(trainBatch[:,3])",np.vstack(trainBatch[:,3]))
                
                #nextPredictedState = self.sess.run(self.modelNet.predict,feed_dict={self.modelNet.scalarInput:trainBatch[:,3],self.modelNet.action:trainBatch[:,1]})[0]
                ##statePredictionErrorBatchArray = self.sess.run(self.modelNet.actionAndState, \
                ##feed_dict={self.modelNet.scalarInput:np.vstack(trainBatch[:,0]),self.modelNet.actions:trainBatch[:,1],self.modelNet.currentState:np.vstack(trainBatch[:,3])})
                ##print("statePredictionErrorBatchArray",statePredictionErrorBatchArray)
                statePredictionErrorBatchArray = self.sess.run(self.modelNet.lossKeepRows, \
                feed_dict={self.modelNet.scalarInput:np.vstack(trainBatch[:,0]),self.modelNet.actions:trainBatch[:,1],self.modelNet.currentState:np.vstack(trainBatch[:,3])})
                #print("statePredictionErrorBatchArray2",statePredictionErrorBatchArray)
                #time.sleep(2.0)
                Q1 = self.sess.run(self.mainQN.predict,feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                Q2 = self.sess.run(self.targetQN.Qout,feed_dict={self.targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                end_multiplier = 1#-(trainBatch[:,4] - 1)
                doubleQ = Q2[range(self.batch_size),Q1]
                targetQ = trainBatch[:,2] + (self.y*doubleQ * end_multiplier) + statePredictionErrorBatchArray*self.exploreRewardConst
                #Update the network with our target values.
                _ = self.sess.run(self.mainQN.updateModel, \
                    feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:,0]),self.mainQN.targetQ:targetQ, self.mainQN.actions:trainBatch[:,1]})
                
                updateTarget(self.targetOps,self.sess) #Update the target network toward the primary network.
            
            if self.total_steps % (self.update_freq_modelNet) == 0:
                trainBatch = self.episodeBuffer.sample(self.batch_size)
                _ = self.sess.run(self.modelNet.updateModel, \
                    feed_dict={self.modelNet.scalarInput:np.vstack(trainBatch[:,0]),self.modelNet.actions:trainBatch[:,1],self.modelNet.currentState:np.vstack(trainBatch[:,3])})
                statePredictionErrorBatchArray = self.sess.run(self.modelNet.lossKeepRows, \
                feed_dict={self.modelNet.scalarInput:np.vstack(trainBatch[:,0]),self.modelNet.actions:trainBatch[:,1],self.modelNet.currentState:np.vstack(trainBatch[:,3])})
                print("statePredictionErrorBatchArray",statePredictionErrorBatchArray)
            if self.total_steps % (self.update_freq*100) == 0:
                for i in self.episodeBuffer.buffer:
                    #print(i[2])
                    if i[2]!= 0 and i[1] == 7:
                        print("found reward and attack move",i)
                #print(trainBatch[:50])
                #time.sleep(2.0)
        #q_predict = #self.q_table.ix[previous_state, previous_action]
        #q_target = #reward + self.gamma * self.q_table.ix[current_state, :].max()

class modelNetwork():
    def __init__(self):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.

        self.currentState = tf.placeholder(shape=[None,8],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,8,dtype=tf.float32)
        self.scalarInput =  tf.placeholder(shape=[None,8],dtype=tf.float32)#(shape=[None,21168],dtype=tf.float32)
        self.actionAndState = tf.concat([self.scalarInput, self.actions_onehot], 1)
        self.fully1 = tf.contrib.layers.fully_connected(self.actionAndState,100)
        self.fully2 = tf.contrib.layers.fully_connected(self.fully1,100)
        
        xavier_init2 = tf.contrib.layers.xavier_initializer()
        self.AW2 = tf.Variable(xavier_init2([100//1,8]))
        self.predict = tf.matmul(self.fully2,self.AW2)
        print("self.predict",self.predict)
        #self.predictNormalised = tf.subtract(self.predict,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predictNormalised = self.predict/tf.reduce_max(tf.abs(self.predict))
        self.currentStateNormalised = self.currentState/tf.reduce_max(tf.abs(self.predict))
        #self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        #self.actions_onehot = tf.one_hot(self.actions,8,dtype=tf.float32)
        #self.predictAction = tf.reduce_sum(tf.multiply(self.predict, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.predictNormalised - self.currentStateNormalised)
        
        self.lossKeepRows = tf.reduce_mean(self.td_error,1)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)#learning rate = 0.0001
        self.updateModel = self.trainer.minimize(self.loss)
        
class Qnetwork():
    def __init__(self,h_size):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        
        self.scalarInput =  tf.placeholder(shape=[None,8],dtype=tf.float32)#(shape=[None,21168],dtype=tf.float32)
        self.fully1 = tf.contrib.layers.fully_connected(self.scalarInput,100)
        self.fully2 = tf.contrib.layers.fully_connected(self.fully1,100)
        '''
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
        '''
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        #print("here",self.fully2.get_shape().as_list())
        self.streamAC,self.streamVC = tf.split(self.fully2,2,1)
        self.streamA = slim.flatten(self.streamAC) # dims was 100, but it got halfed to 50 in the split
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        
        self.AW = tf.Variable(xavier_init([100//2,8]))#dims are 50 by 8.
        self.VW = tf.Variable(xavier_init([100//2,1]))#tf.Variable(xavier_init([h_size//2,1]))
        #print("here2",self.AW.get_shape().as_list(),self.streamA.get_shape().as_list())
        self.Advantage = tf.matmul(self.streamA,self.AW)#matrix mult. happens on 1x50, 50x8 sized arrays resulting in a 1x8 array.
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,8,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)#learning rate = 0.0001
        self.updateModel = self.trainer.minimize(self.loss)
class experience_buffer():
    def __init__(self, buffer_size = 5000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,4])
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def getAvailableSmartActions(available_actions):
    #available_smart_actions = smart_actions
    available_smart_actions = list(range(len(smart_actions)))#copy.deepcopy(smart_actions)
    if 332 not in available_actions:
        del available_smart_actions[7]
    if 477 not in available_actions:
        del available_smart_actions[5]
    if 477 in available_actions:
        del available_smart_actions[4]
    if 42 not in available_actions:
        del available_smart_actions[3]
    if 91 not in available_actions:
        del available_smart_actions[2]
    #print("available_smart_actions",available_smart_actions)
    #print(smart_actions)
    return available_smart_actions
   
'''
ACTION_DO_NOTHING,#0
    ACTION_SELECT_SCV,#1
    ACTION_BUILD_SUPPLY_DEPOT,#2
    ACTION_BUILD_BARRACKS,#3
    ACTION_SELECT_BARRACKS,#4
    ACTION_BUILD_MARINE,#5
    ACTION_SELECT_ARMY,#6
    ACTION_ATTACK#7
'''
   
'''
with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1,r,d = env.step(a)
            s1 = processState(s1)
            
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
            total_steps += 1
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) #Update the target network toward the primary network.
            rAll += r
            s = s1
            
            if d == True:

                break
        
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        #Periodically save the model. 
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
'''
        
"""

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.8):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.counter = 0
        self.actionHistory = []

    def choose_action(self, observation, available_actions):
        print("available_actions",available_actions)
        time.sleep(1.0)
        self.check_state_exist(observation)
        self.counter += 1
        if self.counter%25 == 0:
            print("step_num",self.counter)
            print("observation",observation)
            print("qtable \n",self.q_table.idxmax(axis=1))
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        
        return action

    def learn(self, previous_state, previous_action, reward, current_state):
        self.check_state_exist(current_state)
        self.check_state_exist(previous_state)
        lastActionDidSomething = previous_state != current_state
        
        q_predict = self.q_table.ix[previous_state, previous_action]
        q_target = reward + self.gamma * self.q_table.ix[current_state, :].max()
        if self.counter%25 == 0:
            #print("actionHistory",self.actionHistory)
            print("q_target",q_target)
        # update
        self.q_table.ix[previous_state, previous_action] += self.lr * (q_target - q_predict)
        # need to reward history of steps with exp decrease
        for actionHistoryNum in range(len(self.actionHistory)):
            self.q_table.ix[self.actionHistory[actionHistoryNum][0], self.actionHistory[actionHistoryNum][1]] += self.lr * (q_target - q_predict) * ((actionHistoryNum+1)/len(self.actionHistory))
            
            #if self.counter%25 == 0:
            #    print("reward thing",self.lr * (q_target - q_predict) * ((actionHistoryNum+1)/len(self.actionHistory)))
            #    print(actionHistoryNum)
            #    print(len(self.actionHistory))
            #    print("reward 1",((actionHistoryNum+1)/len(self.actionHistory)))
            
        if len(self.actionHistory)>30:
            self.actionHistory = []
        if lastActionDidSomething:
            self.actionHistory.append([previous_state,previous_action])
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
"""