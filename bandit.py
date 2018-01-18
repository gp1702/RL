import numpy as np
import matplotlib.pyplot as plt


class bandit:
    def __init__(self, k):

        #k arms or actions
        self.k = k
        #set rewards for k actions
        self.true_rewards = np.random.normal(0, 1, k)


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

    def ucb(self, t):
        if(t>1):
            #expected rewards till t-1 for all actions
            Q_a = self.expected_rewards
            #count till t-1 for all actions
            N = self.actioncounts
            #add ucb term to get new expected reward
            Q_a = Q_a + self.c * np.sqrt(np.divide(np.log(t-1), N, where=N!=0))
            #get greedy action and resolve ties randomly
            action = np.random.choice(np.flatnonzero(Q_a == Q_a.max()))
        else:
            action = np.random.choice(np.flatnonzero(self.expected_rewards == self.expected_rewards.max()))

        return action
    def chooseAction(self, t):
        #get action
        if(self.name=="eps-greedy"):
            action = self.epsilongreedy()
        elif(self.name=="ucb"):
            action = self.ucb(t)

        #update action count
        self.actioncounts[action] = self.actioncounts[action] + 1

        return action


    def updateRewards(self, action, reward):
        #self.expected_rewards[action] = (self.expected_rewards[action] * (self.actioncounts[action]-1) + reward) / self.actioncounts[action]
        self.expected_rewards[action] = self.expected_rewards[action] + (reward - self.expected_rewards[action])/self.actioncounts[action]

    def iterate(self, t):
        #choose an action, return arm number
        action = self.chooseAction(t)

        #get the reward of the action
        reward = self.getActionValue(action)

        #update average reward of the system
        self.avg_reward = (self.avg_reward * float(t-1) + reward) / float(t)

        #update the expected reward for the particular action
        self.updateRewards(action, reward)

        return self.avg_reward

    def start(self, timesteps, name, c=0, eps=0.01):
        self.timesteps = timesteps
        self.name = name
        #for ucb
        self.c = float(c)
        #for eps-greedy
        self.eps = float(eps)
        #maintain expected rewards
        self.expected_rewards = np.zeros(self.k)
        self.actioncounts = np.zeros(self.k)
        self.avg_reward = 0.

        avg_reward_overtime = []
        for t in range(1, timesteps+1):
            avg_reward_overtime.append(self.iterate(t))
            print("\r%d"%(t),end="\r")

        self.avg_reward_overtime = avg_reward_overtime

        return avg_reward_overtime


    def plot(self, savedir):
        timesteps = range(1, self.timesteps+1)
        plt.plot(timesteps, self.avg_reward_overtime, label=self.name)
        plt.legend()
        #plt.xscale('log')
        plt.savefig(self.name+'.png', bbox_inches='tight')


#initialize bandit of size 10
trybandit = bandit(k=10)
timesteps = 2000
#run bandit
trybandit.start(timesteps, name="eps-greedy", eps=0.01)
trybandit.plot("plots")
trybandit.start(timesteps, name="ucb", c=2.)
trybandit.plot("plots")
