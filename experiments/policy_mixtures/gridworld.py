from rltools.policy import policy_evaluation, SoftMax_mixture
from itertools import product
from rltools.GridWorld import GridWorld, boundary_condition
import numpy as np
import matplotlib.pyplot as plt
#
# def train(agent, domain, num_episodes):
#     for i in xrange(num_episodes):
#         r_t, s_t = domain.reset()
#         agent.reset()
#         count = 0
#         cumulative_reward = 0
#
#         while s_t != None:
#             # apply an action from the agent
#             # the domain will return a 'None' state when terminating
#             r_t, s_t = domain.step(agent.step(r_t, s_t))
#             count += 1
#             cumulative_reward += r_t
#
#         # final update step for the agent
#         agent.step(r_t, s_t)
#
#         if i % 2 == 0:
#             if render_value_fn:
#                 plt.gca().clear()
#                 plt.contourf(*getValueFn(valuefn))
#                 plt.title('episode ' + str(i))
#                 plt.savefig(file_path + str(k) + '.png')
#                 k +=1
#

gamma = 0.9

pis = [lambda s: 0, lambda s: 1, lambda s: 2, lambda s: 3]
middle = set()
for x in product(range(4,7), range(4,7)):
    s = np.array(x, dtype='int32')
    s.flags.writeable = False
    middle.add(s.data)

def reward(state, action=None):
    s = state.copy()
    s.flags.writeable = False
    if s.data in middle:
        return 1
    else:
        return 0

start_range = [np.zeros(2, dtype = 'int32'), np.ones(2, dtype = 'int32')*9]
boundary = boundary_condition(start_range)

islegal = lambda s: True

def terminal(state):
    return not boundary(state) #np.any(state < 0) or  np.any(state >9)

class phi(object):
    def __init__(self):
        self.size = 12*12+1
    def __call__(self, state):
        phi_t = np.zeros(self.size, dtype = 'int32')
        if phi_t != None:
            phi_t[np.ravel_multi_index(state+1, (12,12))] = 1
            phi_t[-1] = 1
        return phi_t


dx = np.array([ [1,0],
                [-1,0],
                [0,1],
                [0,-1]], dtype='int32')


gridworld = GridWorld(reward,
                      islegal,
                      terminal,
                      start_range,
                      random_start = True,
                      max_episode = 20)
valuefns = [policy_evaluation([reward],
                              gamma,
                              pi,
                              gridworld,
                              projector = phi(),
                              method = 'LSTDlambda',
                              number_episodes = 1000,
                              max_episode_length = 20)[-1]
            for pi in pis]
pi_mix = SoftMax_mixture(valuefns, pis)
value_mix = [policy_evaluation([reward],
                              gamma,
                              pi_mix,
                              gridworld,
                              projector = phi(),
                              method = 'LSTDlambda',
                              number_episodes = 2000,
                              max_episode_length = 20)[-1]]

val =[np.zeros((10,10)) for i in range(5)]
for x,y in product(range(10), range(10)):
    state = np.array([x,y], dtype='int32')
    for v, fn in zip(val, valuefns+value_mix):
        v[x,y] = fn(state)

fig, axes = plt.subplots(1, len(val))
for ax, v, i in zip(axes, val, range(len(val))):
    ax.imshow(v, interpolation = 'none')
    ax.set_title(str(i) if i != 4 else 'mixture')

plt.show()
