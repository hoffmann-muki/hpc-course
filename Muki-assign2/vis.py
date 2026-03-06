#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sys

# Usage: python3 vis.py <path_to_file>
assert(len(sys.argv) == 2)
plt.imshow(np.genfromtxt(sys.argv[1], delimiter=','),
           cmap='coolwarm',
           interpolation='nearest')
plt.show()
