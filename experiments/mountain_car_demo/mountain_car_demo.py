from rltools.MountainCar import MountainCar
from rltools.representation import TileCodingDense, TileCoding
from rltools.valuefn import LinearTD
from rltools.policy import Egreedy
from rltools.agent import TabularActionSarsa
from rltools.valuefn import LSTDlambda
from rltools.experiment import evaluate_trial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from itertools import product
import os

domain = MountainCar(random_start= False, max_episode=10000)

act = domain.discrete_actions


proj = TileCoding(input_indicies = [np.array([0,1])],
                 ntiles = [12],
                 ntilings=[10],
                 hashing=None,
                 state_range = domain.state_range,
                 bias_term = True)
class PHI(object):
    def __init__(self):
        self.size = proj.size
    def __call__(self, s):
        return proj(s) if s != None else np.zeros(proj.size)

phi = PHI()

valuefn = LinearTD(len(domain.discrete_actions),
                   phi,
                   alpha = 0.01,
                   lamb = 0.0,
                   gamma= 0.99)
policy = Egreedy(np.arange(len(domain.discrete_actions)), valuefn, epsilon = 0.05)
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)


num_episodes = 500

def eval(valuefn):
    val = np.empty(10000)
    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], 100)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], 100)
    for i,s in enumerate(product(x, y)):
        s = np.array(s)
        val[i] = max([valuefn(s,0), valuefn(s,1), valuefn(s,2)])
    X, Y = np.meshgrid(x, y)
    return X, Y, val.reshape((100,100)).T


if not os.path.exists("mountain_car_demo_figures"):
    os.system("mkdir mountain_car_demo_figures")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.view_init(55,55)
ax.plot_surface(*eval(valuefn), cmap = cm.coolwarm)
plt.contourf(*eval(valuefn))
plt.title('initial')
plt.savefig('mountain_car_demo_figures/0.png')


k = 1
for i in xrange(num_episodes):
    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    while s_t != None:
        r_t, s_t = domain.step(agent.step(r_t, s_t))
        count += 1
    agent.step(r_t, s_t)
 
    if i % 25 == 0:
        plt.gca().clear()
        plt.gcf().clear()
        plt.contourf(*eval(valuefn))
        plt.colorbar()
        fig.clear()
        ax = fig.gca(projection='3d')
        ax.view_init(55,55)
        ax.plot_surface(*eval(valuefn), cmap = cm.coolwarm)
        plt.title('episode ' + str(i))
        plt.savefig('mountain_car_demo_figures/' + str(k) + '.png')
        k +=1
        #plt.pause(0.0005)
        
    result =evaluate_trial(domain, agent)
    print "current policy yields: %s, learning episode step count: %d" \
            % (result, count)

