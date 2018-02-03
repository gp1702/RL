import numpy as np
import matplotlib.pyplot as plt
import os

class bandit:
    def __init__(self, k, variance=1, seed=123):

        #k arms or actions
        self.k = k
        
        #set rewards for k actions
        np.random.seed(seed=seed)
        self.true_rewards = np.random.normal(0, variance, k)

        self.bestaction = np.random.choice(np.flatnonzero(self.true_rewards == self.true_rewards.max()))

    def getActionValue(self, action):
        #get action reward using normal dist with mean as true value and variance 1
        return np.random.normal(self.true_rewards[action], 1, 1)[0]

    def epsilongreedy(self):
        eps = np.random.random()
        #epsilon greedy method
        if(eps>self.eps):
            #get greedy action and resolve ties randomly
            N = self.alphas + self.betas
            expected_rewards = np.divide(self.alphas, N, where=N!=0)

            action = np.random.choice(np.flatnonzero(expected_rewards == expected_rewards.max()))
        else:
            #get random action
            action = np.random.choice(self.k)

        return action

    def ucb(self, t):
        #count till t-1 for all actions
        N = self.alphas + self.betas
        #expected rewards till t-1 for all actions
        Q_a = np.divide(self.alphas, N, where=N!=0)
        Q_a = Q_a + self.c * np.sqrt(np.divide(np.log(t), N+1.))

        #Q_a = Q_a + self.c * np.sqrt(np.divide(np.log(t-1), N, where=N!=0))
        #get greedy action and resolve ties randomly
        action = np.random.choice(np.flatnonzero(Q_a == Q_a.max()))
        #print(N)
        return action

    def thompson(self):
        N = self.alphas + self.betas
        expected_rewards = np.divide(self.alphas, N, where=N!=0)
        #sample rewards
        sampled_rewards = np.random.normal(expected_rewards, 1./(N+1.))
        #pick greedy action
        action = np.random.choice(np.flatnonzero(sampled_rewards == sampled_rewards.max()))

        return action

    def chooseAction(self, t):
        #get action
        if(self.name=="eps-greedy"):
            action = self.epsilongreedy()
        elif(self.name=="ucb"):
            action = self.ucb(t)
        elif(self.name=="thompson"):
            action = self.thompson()

        #update action count
        self.actioncounts[action] = self.actioncounts[action] + 1

        return action


    def updateRewards(self, action, reward):
        #self.expected_rewards[action] = (self.expected_rewards[action] * (self.actioncounts[action]-1) + reward) / self.actioncounts[action]
        self.expected_rewards[action] = self.expected_rewards[action] + (reward - self.expected_rewards[action])/self.actioncounts[action]
        self.alphas[action] += reward
        self.betas[action] += 1. - reward

    def iterate(self, t):
        #choose an action, return arm number
        action = self.chooseAction(t)

        #get the reward of the action
        reward = self.getActionValue(action)

        #update average reward of the system
        self.avg_reward = (self.avg_reward * float(t-1) + reward) / float(t)

        #update the expected reward for the particular action
        self.updateRewards(action, reward)

        #calculate regret
        regret = self.true_rewards[self.bestaction] - self.true_rewards[action]
        self.regret = (self.regret * float(t-1) + regret) / float(t)

        optimal_reward = int(self.bestaction==action)
        self.optimal_reward += optimal_reward

        return self.regret

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
        #successes
        self.alphas = np.zeros(self.k)
        #failures
        self.betas = np.zeros(self.k)
        self.avg_reward = 0.
        self.regret = 0.
        self.optimal_reward = 0

        regret_overtime = []
        optimal_reward = []
        for t in range(1, timesteps+1):
            regret_overtime.append(self.iterate(t))
            optimal_reward.append(float(self.optimal_reward)/float(t))
            #print("\r%d"%(t),end="\r")

        self.regret_overtime = regret_overtime

        return optimal_reward


    def plot(self, savedir, noshow=False, ylim=None):
        timesteps = range(1, self.timesteps+1)
        plt.plot(timesteps, self.regret_overtime, label=self.name)
        plt.legend()
        #plt.xscale('log')
        if(noshow is False):
            if(ylim is not None):
                plt.ylim(ymax=ylim)
            plt.xlabel('Time Steps')
            plt.ylabel('Regret')
            plt.show()
            #plt.savefig(os.path.join(savedir, self.name+'.png'), bbox_inches='tight')


#initialize bandit of size 10
#trybandit = bandit(k=5)
#timesteps = 1000
#run bandit
#trybandit.start(timesteps, name="eps-greedy", eps=0.01)
#trybandit.plot("plots", noshow=True)
#trybandit.start(timesteps, name="ucb", c=2.)
#trybandit.plot("plots", noshow=True)
#trybandit.start(timesteps, name="thompson")
#trybandit.plot("plots")
