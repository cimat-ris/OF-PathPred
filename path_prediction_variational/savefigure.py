import os
import matplotlib as mpl
if os.environ.get('DISPLAY', '' ) == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

x = np.array(range(100))
y = x**2

plt.plot(x,y)
plt.savefig('save_fig.png')
