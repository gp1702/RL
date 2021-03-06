# coding: utf-8

# import libraries
import gym
import numpy as np


# transforms a deterministic policy in a policy matrix
def vectorize_policy(policy, nS, nA):
    new_policy = np.zeros([nS, nA])
    for s in range(nS):
        new_policy[s, policy[s]] = 1.0
    return new_policy


# calculate the mean reward received for each state under the current policy
def calculate_mean_reward(R, T, policy):
    if (len(policy.shape) == 1):
        nS, nA, nS = T.shape
        policy = vectorize_policy(policy, nS, nA)

    return np.einsum('ijk,ijk,ij ->i', R, T, policy)


# calculate the transition probability under the given policy
def calculate_mean_transition(T, policy):
    if (len(policy.shape) == 1):
        nS, nA, nS = T.shape
        policy = vectorize_policy(policy, nS, nA)

    return np.einsum('ijk,ij -> ik', T, policy)


# solve Bellman equation through iteration method
def solve_Bellman_iteration(R, T, policy, k=1000, gamma=1.0):
    """
    This function finds the value function of the current policy by successive
    iterations of the Bellman Equation.

    --input--
    R: rewards for every transition
    T: transition probabilities
    policy: initial policy
    k: maximum number of iterations
    gamma: discount factor (< 1 for convergence guarantees)

    """

    # calculate mean reward and the mean transition matrix
    mean_R = calculate_mean_reward(R, T, policy)
    mean_T = calculate_mean_transition(T, policy)

    # initializes value function to 0
    value_function = np.zeros(mean_R.shape)

    # iterate k times the Bellman Equation
    for i in range(k):
        value_function = mean_R + gamma * np.dot(mean_T, value_function)

    return value_function


# find optimal policy through POLICY ITERATION algorithm
def policy_iteration(R, T, policy, max_iter=100, k=100, gamma=1.0):
    nS, nA, nS = T.shape
    opt = np.zeros(nS)

    for _ in range(max_iter):
        # store current policy
        opt = policy.copy()

        # evaluate value function (at least approximately)
        v = solve_Bellman_iteration(R, T, policy, k, gamma)

        # calculate q-function
        q = np.einsum('ijk,ijk->ij', T, R + gamma * v[None, None, :])
        print("Value-Function:", v)
        print("Action-Value-Function:", q)

        # update policy
        policy = np.argmax(q, axis=1)
        print("Policy",policy)

        # if policy did not change, stop
        if np.array_equal(policy, opt):
            break

    return policy


def value_iteration(env, gamma, R, T, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    See page 90 (pg 108 pdf) of the Sutton and Barto Second Edition
    book.
    http://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.
    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """

    value_func = np.zeros(env.nS)

    for i in range(max_iterations):
      # shape = (nS, nA)
      prev_value_func = value_func.copy()
      q = np.einsum("ijk,ijk -> ij", env.T, env.R + gamma * value_func)
      value_func = np.max(q, axis=1)

      if np.max(np.abs(value_func-prev_value_func)) < tol:
        break

    return value_func, i


# start environment.
env = gym.make('FrozenLake-v0')
nA, nS = env.env.nA, env.env.nS

# reward and transition matrices
T = np.zeros([nS, nA, nS])
R = np.zeros([nS, nA, nS])
for s in range(nS):
    for a in range(nA):
        transitions = env.env.P[s][a]
        for p_trans, next_s, rew, done in transitions:
            T[s, a, next_s] += p_trans
            R[s, a, next_s] = rew
        T[s, a, :] /= np.sum(T[s, a, :])

# calculate optimal policy
policy = (1.0 / nA) * np.ones([nS, nA])  # initilize policy randomly
opt = policy_iteration(R, T, policy, max_iter=10000, k=100, gamma=0.9999)

# test optimal policy
max_time_steps = 100000
n_episode = 1

#env.monitor.start('./frozenlake-experiment', force=True)

for i_episode in range(n_episode):

    observation = env.reset()  # reset environment to beginning

    # run for several time-steps
    for t in range(max_time_steps):
        # display experiment
        # env.render()

        # sample a random action
        action = opt[observation]

        # observe next step and get reward
        observation, reward, done, info = env.step(action)

        if done:
            env.render()
            print
            "Simulation finished after {0} timesteps".format(t)
            break

#env.monitor.close()


#gym.upload('/home/lucianodp/Documents/eua/reinforcement_learning/notebooks/frozenlake-experiment',
           #api_key='sk_qkx3jhBbTRamxadtXqA3pQ')
