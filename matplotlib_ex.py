'''
练习使用matplotlib

出现了小插曲，当这个文件命名为matplotlib.py时，出现奇怪的错误
后来发现是与已有的库文件名重复
可能与python的文件引用有关吧
'''

import numpy as np
import matplotlib.pyplot as plt
from time import sleep
# exercise 1：简单实用
plt.ion()
x = np.arange(-np.pi, np.pi, 0.1)
y = np.sin(x)
plt.plot(x, y, 'g')
plt.draw()
plt.show()
sleep(2)