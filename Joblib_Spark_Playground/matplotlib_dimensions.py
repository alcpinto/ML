import matplotlib.pyplot as plt
import numpy as np

#fake example data
song1 = np.asarray([1, 2, 3, 4, 5, 6, 2, 35, 4, 1])
song2 = song1*2
song3 = song1*1.5

#list of arrays containing all data
data = [song1, song2, song3]

#calculate 2d indicators
def indic(data):
    #alternatively you can calulate any other indicators
    max = np.max(data, axis=1)
    min = np.min(data, axis=1)
    return max, min

x,y = indic(data)
plt.scatter(x, y, marker='x')
plt.show()

import pandas as pd
pd.DataFrame(data).T.plot()
plt.show()


# More examples:
# https://www.python-graph-gallery.com/
