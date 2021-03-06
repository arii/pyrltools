from rltools.representation import TileCodingDense, UNH, PythonHash, IdentityHash
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import timeit

state_range = [ np.array([-10,-10]), np.array([10, 10])]
phi_1 = TileCodingDense([[0,1]],
                   [10],
                   [20],
                   None,
                   state_range,
                   bias_term = True)
phi_2 = TileCodingDense([[0,1]],
                   [20],
                   [10],
                   [PythonHash(200)],
                   state_range,
                   bias_term = True)
phi_3 = TileCodingDense([[0,1]],
                   [20],
                   [10],
                   [UNH(200)],
                   state_range,
                   bias_term = True)
print phi_1.size
print phi_2.size
print phi_3.size

resolution = 100
grid = [np.array(x) for x in product(np.linspace(-10, 10, resolution),
                                             np.linspace(-10, 10, resolution))]
Phi_1 = np.array([phi_1(x) for x in grid])
Phi_2 = np.array([phi_2(x) for x in grid])
Phi_3 = np.array([phi_3(x) for x in grid])

# print timeit.timeit('[phi_1(x) for x in grid]', setup="from __main__ import grid, phi_1", number=10)
# print timeit.timeit('[phi_2(x) for x in grid]', setup="from __main__ import grid, phi_2", number=10)
# print timeit.timeit('[phi_3(x) for x in grid]', setup="from __main__ import grid, phi_3", number=10)

def test_and_plot(f):
    y = np.array([f(x) for x in grid])

    res_1 = np.linalg.lstsq(Phi_1, y)
    res_2 = np.linalg.lstsq(Phi_2, y)
    res_3 = np.linalg.lstsq(Phi_3, y)

    fig, axes = plt.subplots(2,2)
    faxes = axes.flat
    faxes[0].imshow(Phi_1.dot(res_1[0]).reshape(resolution,resolution), interpolation = 'none')
    faxes[1].imshow(Phi_2.dot(res_2[0]).reshape(resolution,resolution), interpolation = 'none')
    faxes[2].imshow(Phi_3.dot(res_3[0]).reshape(resolution,resolution), interpolation = 'none')
    faxes[3].bar(np.arange(3),
                 [np.linalg.norm(Phi_1.dot(res_1[0]) - y),
                 np.linalg.norm(Phi_2.dot(res_2[0]) - y),
                 np.linalg.norm(Phi_3.dot(res_3[0]) - y)])
    plt.show()

f3 = lambda x: np.sum(x**3) + 10
f2 = lambda x: x.dot(x) + 10
f1 = lambda x: np.sum(x) + 10
f4 = lambda x: np.sum(np.sin(x)) + 2

test_and_plot(f1)
test_and_plot(f2)
test_and_plot(f3)
test_and_plot(f4)