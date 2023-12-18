#!/usr/bin/env python3
# -*- coding: utf-8 -*-import numpy as np

import matplotlib.pyplot as plt

def f(x):
    return np.exp(np.sin(x))

def df(x):
    return np.cos(x) * np.exp(np.sin(x))

x = np.linspace(-np.pi, np.pi, 100)

plt.plot(x, f(x), label='f(x) = e^sin(x)')

tgx = np.pi/4
tgy = f(tgx)
tgslope = df(tgx)
tg = tgy + tgslope*(x - tgx)

plt.plot(x, tg, label='Tg')
plt.scatter(tgx, tgy, color='red', label='Точка касания')
plt.title('График функции f(x) = e^sin(x) и её касательной')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.annotate('Точка касания', xy=(tgx, tgy), xytext=(tgx-2, tgy+0.2), arrowprops=dict(facecolor='black', arrowstyle='->'))
plt.show()