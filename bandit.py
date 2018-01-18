import numpy as np
import matplotlib.pyplot as plt


class bandit:
    def __init__(self, k, eps=0.01):
        #set epsilon will be used only for epsilon-greedy method
        self.eps = eps

        #k arms or actions
        self.k = k
        #set rewards for k actions
        self.true_rewards = np.random.normal(0, 1, k)
        #maintain expected rewards
        self.expected_rewards = np.zeros(k)
        self.actioncounts = np.zeros(k)
        self.avg_reward = 0.

    def getActionValue(self, action):
        #get action reward using normal dist with mean as true value and variance 1
        return np.random.normal(self.true_rewards[action], 1, 1)[0]

    def epsilongreedy(self):
        eps = np.random.random()
        #epsilon greedy method
        if(eps>self.eps):
            #get greedy action and resolve ties randomly
            action = np.random.choice(np.flatnonzero(self.expected_rewards == self.expected_rewards.max()))
        else:
            #get random action
            action = np.random.choice(3)

        return action

    def chooseAction(self):
        #get action
        action = self.epsilongreedy()
        #update action count
        self.actioncounts[action] = self.actioncounts[action] + 1

        return action


    def updateRewards(self, action, reward):
        #self.expected_rewards[action] = (self.expected_rewards[action] * (self.actioncounts[action]-1) + reward) / self.actioncounts[action]
        self.expected_rewards[action] = self.expected_rewards[action] + (reward - self.expected_rewards[action])/self.actioncounts[action]

    def iterate(self, t):
        #choose an action, return arm number
        action = self.chooseAction()

        #get the reward of the action
        reward = self.getActionValue(action)

        #update average reward of the system
        self.avg_reward = (self.avg_reward * float(t-1) + reward) / float(t)

        #update the expected reward for the particular action
        self.updateRewards(action, reward)

        return self.avg_reward

    def start(self, timesteps, name):
        self.timesteps = timesteps
        self.name = name

        avg_reward_overtime = []
        for t in range(1, timesteps+1):
            avg_reward_overtime.append(self.iterate(t))
            print("\r%d"%(t),end="\r")

        self.avg_reward_overtime = avg_reward_overtime


    def plot(self, savedir):
        timesteps = range(1, self.timesteps+1)
        plt.plot(timesteps, self.avg_reward_overtime)
        plt.xscale('log')
        plt.savefig(self.name+'.png', bbox_inches='tight')


#initialize bandit of size 10
trybandit = bandit(k=10)
#run bandit
trybandit.start(1000, name="eps-greedy")
trybandit.plot("plots")
