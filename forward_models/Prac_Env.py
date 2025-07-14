#Practice Environment
#This will serve as a practice kitchen for cooking up custom environments
#The goal is create an environment that is modeled as a logarithmic variable resistor
import gym
from gym import spaces
import numpy as np
from enum import Enum
import random
from gym.utils import seeding
import serial
import time

class DummySerial:
    def write(self, data):
        print(f"[DummySerial] Would send: {data.decode().strip()}")

    def close(self):
        print("[DummySerial] Closed")

class prac_env_v0(gym.Env):
    def __init__(self, test_mode=False):
        super(prac_env_v0, self).__init__()

        self.test_mode = test_mode
        self.prev_action = 0.0
        #Observation: [current_state, target_current, delta _to_target, prev_action]
        self.current_state_min = -375.0
        self.target_current_min = -275.0
        self.delta_to_target_min = np.abs(self.current_state_min - self.target_current_min)
        self.prev_action_min = -100
        self.low = np.array([self.current_state_min, self.target_current_min,
                              self.delta_to_target_min, self.prev_action_min], dtype=np.float32)
        self.current_state_max = 375.0
        self.target_current_max = 275.0
        self.delta_to_target_max = np.abs(self.current_state_max - self.target_current_max)
        self.prev_action_max = -100
        self.high = np.array([self.current_state_max, self.target_current_max,
                              self.delta_to_target_max, self.prev_action_max], dtype=np.float32)

        #Shape of observation space
        self.observation_space = spaces.Box( low = self.low, high = self.high, dtype = float)
        
        #Possible actions
        self.action_list = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100,
                            -0.0001, -0.0003, -0.001, -0.003, -0.01, -0.03, -0.1, -0.3,  -1, -3, -10, -30, -100]

        #Action space, how the environment will interpret the actions
        self.action_space = spaces.Discrete(len(self.action_list))
        # print(f"Debug: {self.action_space}")
        
        # Set up serial port or dummy serial
        if not self.test_mode:
            try:
                self.ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
                time.sleep(2)  # Allow time for Arduino boot-up
                print("[Serial] Connection to hardware established.")
            except Exception as e:
                print("[Serial] Failed to connect:", e)
                self.ser = None
        else:
            self.ser = DummySerial()
            print("[Test Mode] Using DummySerial")

    def seed(self, seed=None):
        
        #Seed is used to generate random numbers, used for reproducibility
        self.np_random, seed = seeding.np_random(seed)

        #Everytime we run algorithm, we generate 1 new seed. This seed generates a random current,
        #which does not change for the running of the code. It does change everytime we run the code.
        self.current = self.np_random.uniform(-275,275)
        return [seed]

    #reset environment
    def reset(self):

        if hasattr(self, 'state'): print(self.state, self.current)

        #Reseting system will give the inital state of the system a random state
        self.state = self.np_random.uniform(-375.0,375.0) #May need some range
        #Todo: randomize self.current
        #Reseting reward
        self.cumulative_reward = 0
        self.prev_action = 0.0

        print("=================RESET!!==================")
        obs  = np.array([self.state, self.current, self.state-self.current,self.prev_action],dtype = np.float32)

        return obs
    
    #time step which contains most of the logic used in the environment
    #It accepts an action, computes the state of environment after applying that action and returns the tuple (self.state, reward)
    def step(self,action):

        _ = {}
        done = False

        #initial reward
        reward = -0.001

        #current and initial state of system
        current = self.current
        current_state = self.state

        #Where algorithm actually makes its decision
        direction = self.action_list[action]

        #Resulting state due to action
        next_state = current_state + direction

        # If in test mode, use dummy serial
        # Send command to hardware or dummy
        if self.ser:
            try:
                command = f"{direction:.6f}\n"
                self.ser.write(command.encode('utf-8'))
                print(f"[Action] Sent to hardware: {command.strip()}")
            except Exception as e:
                print("[Serial Error]", e)

        #reward system
        if np.abs(next_state) > 374:
            self.cumulative_reward = -10
            done = True

        if np.abs(next_state-current) > np.abs(current_state-current):
            self.cumulative_reward = -1

        if np.abs(next_state-current) < np.abs(current_state-current):
            self.cumulative_reward = 1

        if current == next_state:
            self.state = next_state
            self.prev_action = direction
            self.cumulative_reward = 10
            done = True

        if not done:
            self.state = next_state
            self.prev_action = direction
            print(f"[Not Done] Current state: {self.state}, Action: {direction}")
        #Update rewards
        reward = self.cumulative_reward + reward
        print(f"[Reward] Cumulative reward: {reward}")
        #Done is when you reach the target Current (in our case)

        obs = np.array([self.state, self.current, self.state-self.current, self.prev_action], dtype=np.float32)

        if done:
            print(f"[Done] Final state: {self.state}")
            print("---------------------DONE------------------------")

        return obs, reward, done, _

    #Inteface output 
    def render(self, mode = 'human'):
        print(f'State: {self.state}')
    
    #close any open resources used by environment
    def close(self):
        #pass
        if self.ser:
            self.ser.close()
            print("[Serial] Connection closed.")
