#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 1
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
plt.show()

# 2
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
y = x

plt.title('Линейная зависимость y = x')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.plot(x, y)
plt.show()

# 3
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
y = x

plt.title('Линейная зависимость y = x')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.plot(x, y, 'r--')
plt.show()

# 4
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
y1 = x
y2 = [i**2 for i in x]

plt.title('Зависимости: y1 = x, y2 = x^2')
plt.xlabel('x')
plt.ylabel('y1, y2')
plt.grid()
plt.plot(x, y1, x, y2)
plt.show()

# 5
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 50)
y1 = x
y2 = [i**2 for i in x]

plt.figure(figsize=(9, 9))
plt.subplot(2, 1, 1)
plt.plot(x, y1)
plt.title('Зависимости: y1 = x, y2 = x^2')
plt.ylabel('y1', fontsize=14)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x, y2)
plt.xlabel('x', fontsize=14)
plt.ylabel('y2', fontsize=14)
plt.grid(True)
plt.show()

# 6
import matplotlib.pyplot as plt

fruits = ['apple', 'peach', 'orange', 'bannana', 'melon']
counts = [34, 25, 43, 31, 17]
plt.bar(fruits, counts)
plt.title('Fruits!')
plt.xlabel('Fruit')
plt.ylabel('Count')
plt.show()

# 7
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
AutoMinorLocator)
import numpy as np

x = np.linspace(0, 10, 10)
y1 = 4*x
y2 = [i**2 for i in x]
fig, ax = plt.subplots(figsize=(8, 6))

ax.set_title('Графики зависимостей: y1=4*x, y2=x^2', fontsize=16)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y1, y2', fontsize=14)
ax.grid(which='major', linewidth=1.2)
ax.grid(which='minor', linestyle='--', color='gray', linewidth=0.5)
ax.scatter(x, y1, c='red', label='y1 = 4*x')
ax.plot(x, y2, label='y2 = x^2')
ax.legend()
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='major', length=10, width=2)
ax.tick_params(which='minor', length=5, width=1)
plt.show()

# 8
import matplotlib.pyplot as plt

plt.plot([1, 7, 3, 5, 11, 1])
plt.show()

# 9
import matplotlib.pyplot as plt

plt.plot([1, 5, 10, 15, 20], [1, 7, 3, 5, 11])
plt.show()

# 10
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y = [1, 7, 3, 5, 11]

plt.plot(x, y, label='steel price')
plt.title('Chart price', fontsize=15)
plt.xlabel('Day', fontsize=12, color='blue')
plt.ylabel('Price', fontsize=12, color='blue')
plt.legend()
plt.grid(True)
plt.text(15, 4, 'grow up!')
plt.show()

# 11.1
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y = [1, 7, 3, 5, 11]

plt.plot(x, y, '--')
plt.show()

# 11.2
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y = [1, 7, 3, 5, 11]
line = plt.plot(x, y)

plt.setp(line, linestyle='--')
plt.show()

# 12.1
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [i*1.2 + 1 for i in y1]
y3 = [i*1.2 + 1 for i in y2]
y4 = [i*1.2 + 1 for i in y3]

plt.plot(x, y1, '-', x, y2, '--', x, y3, '-.', x, y4, ':')
plt.show()

# 12.2
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [i*1.2 + 1 for i in y1]
y3 = [i*1.2 + 1 for i in y2]
y4 = [i*1.2 + 1 for i in y3]

plt.plot(x, y1, '-')
plt.plot(x, y2, '--')
plt.plot(x, y3, '-.')
plt.plot(x, y4, ':')
plt.show()

# 13
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y = [1, 7, 3, 5, 11]

plt.plot(x, y, '--r')
plt.show()

# 14
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y = [1, 7, 3, 5, 11]

plt.plot(x, y, 'ro')
plt.show()

# 15
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y = [1, 7, 3, 5, 11]

plt.plot(x, y, 'bx')
plt.show()

# 16.1
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [i*1.2 + 1 for i in y1]
y3 = [i*1.2 + 1 for i in y2]
y4 = [i*1.2 + 1 for i in y3]

plt.figure(figsize=(12, 7))

plt.subplot(2, 2, 1)
plt.plot(x, y1, '-')
plt.subplot(2, 2, 2)
plt.plot(x, y2, '--')
plt.subplot(2, 2, 3)
plt.plot(x, y3, '-.')
plt.subplot(2, 2, 4)
plt.plot(x, y4, ':')
plt.show()

# 16.2 
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [i*1.2 + 1 for i in y1]
y3 = [i*1.2 + 1 for i in y2]
y4 = [i*1.2 + 1 for i in y3]

plt.figure(figsize=(12, 7))

plt.subplot(221)
plt.plot(x, y1, '-')
plt.subplot(222)
plt.plot(x, y2, '--')
plt.subplot(223)
plt.plot(x, y3, '-.')
plt.subplot(224)
plt.plot(x, y4, ':')
plt.show()

# 16.3
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [i*1.2 + 1 for i in y1]
y3 = [i*1.2 + 1 for i in y2]
y4 = [i*1.2 + 1 for i in y3]

fig, axs = plt.subplots(2, 2, figsize=(12, 7))

axs[0, 0].plot(x, y1, '-')
axs[0, 1].plot(x, y2, '--')
axs[1, 0].plot(x, y3, '-.')
axs[1, 1].plot(x, y4, ':')

plt.show()

# 17.1
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [4, 3, 1, 8, 12]

plt.plot(x, y1, 'o-r', label='line 1')
plt.plot(x, y2, 'o-.g', label='line 1')
plt.legend()
plt.show()

# 17.2
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [4, 3, 1, 8, 12]

plt.plot(x, y1, 'o-r')
plt.plot(x, y2, 'o-.g')
plt.legend(['L1', 'L2'])
plt.show()

# 18
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [4, 3, 1, 8, 12]

line1, = plt.plot(x, y1, 'o-b')
line2, = plt.plot(x, y2, 'o-.m')

plt.legend((line2, line1), ['L2', 'L1'])
plt.show()

# 19
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [4, 3, 1, 8, 12]
locs = ['best', 'upper right', 'upper left', 'lower left',
        'lower right', 'right', 'center left', 'center right',
        'lower center', 'upper center', 'center']

plt.figure(figsize=(12, 12))

for i in range(3):
    for j in range(4):
        if i*4+j < 11:
            plt.subplot(3, 4, i*4+j+1)
            plt.title(locs[i*4+j])
            plt.plot(x, y1, 'o-r', label='line 1')
            plt.plot(x, y2, 'o-.g', label='line 2')
            plt.legend(loc=locs[i*4+j])
            plt.show()
        else:
            break

# 20 
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [4, 3, 1, 8, 12]

plt.plot(x, y1, 'o-r', label='line 1')
plt.plot(x, y2, 'o-.g', label='line 1')
plt.legend(bbox_to_anchor=(1, 0.6))
plt.show()

# 21
import matplotlib.pyplot as plt

x = [1, 5, 10, 15, 20]
y1 = [1, 7, 3, 5, 11]
y2 = [4, 3, 1, 8, 12]

plt.plot(x, y1, 'o-r', label='line 1')
plt.plot(x, y2, 'o-.g', label='line 1')
plt.legend(fontsize=14, shadow=True, framealpha=1, facecolor='y', edgecolor='r', title='Легенда')
plt.show()

# 22
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = [9, 4, 2, 4, 9]
y2 = [1, 7, 6, 3, 5]

fg = plt.figure(figsize=(7, 3), constrained_layout=True)
gs = gridspec.GridSpec(ncols=2, nrows=1, figure=fg)

fig_ax_1 = fg.add_subplot(gs[0, 0])
plt.plot(x, y1)

fig_ax_2 = fg.add_subplot(gs[0, 1])
plt.plot(x, y2)

plt.show()

# 23
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y1 = [9, 4, 2, 4, 9]
y2 = [1, 7, 6, 3, 5]
y3 = [-7, -4, 2, -4, -7]

fg = plt.figure(figsize=(9, 4), constrained_layout=True)
gs = fg.add_gridspec(2, 2)

fig_ax_1 = fg.add_subplot(gs[0, :])
plt.plot(x, y2)

fig_ax_2 = fg.add_subplot(gs[1, 0])
plt.plot(x, y1)

fig_ax_3 = fg.add_subplot(gs[1, 1])
plt.plot(x, y3)

plt.show()

# 24
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

fg = plt.figure(figsize=(9, 9), constrained_layout=True)
gs = fg.add_gridspec(5, 5)

fig_ax_1 = fg.add_subplot(gs[0, :3])
fig_ax_1.set_title('gs[0, :3]')
fig_ax_2 = fg.add_subplot(gs[0, 3:])
fig_ax_2.set_title('gs[0, 3:]')
fig_ax_3 = fg.add_subplot(gs[1:, 0])
fig_ax_3.set_title('gs[1:, 0]')
fig_ax_4 = fg.add_subplot(gs[1:, 1])
fig_ax_4.set_title('gs[1:, 1]')
fig_ax_5 = fg.add_subplot(gs[1, 2:])
fig_ax_5.set_title('gs[1, 2:]')
fig_ax_6 = fg.add_subplot(gs[2:4, 2])
fig_ax_6.set_title('gs[2:4, 2]')
fig_ax_7 = fg.add_subplot(gs[2:4, 3:])
fig_ax_7.set_title('gs[2:4, 3:]')
fig_ax_8 = fg.add_subplot(gs[4, 3:])
fig_ax_8.set_title('gs[4, 3:]')

plt.show()

# 25
import matplotlib.pyplot as plt

fg = plt.figure(figsize=(5, 5),constrained_layout=True)
widths = [1, 3]
heights = [2, 0.7]
gs = fg.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)

fig_ax_1 = fg.add_subplot(gs[0, 0])
fig_ax_1.set_title('w:1, h:2')
fig_ax_2 = fg.add_subplot(gs[0, 1])
fig_ax_2.set_title('w:3, h:2')
fig_ax_3 = fg.add_subplot(gs[1, 0])
fig_ax_3.set_title('w:1, h:0.7')
fig_ax_4 = fg.add_subplot(gs[1, 1])
fig_ax_4.set_title('w:3, h:0.7')

plt.show()

# 26
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.figtext(0.5, -0.1, 'figtext')
plt.suptitle('suptitle')
plt.subplot(121)
plt.title('title')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.text(0.2, 0.2, 'text')
plt.annotate('annotate', xy=(0.2, 0.4), xytext=(0.6, 0.7), arrowprops=dict(facecolor='black', shrink=0.05))
plt.subplot(122)
plt.title('title')
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.text(0.5, 0.5, 'text')
plt.show()

# 27
import matplotlib.pyplot as plt

weight=['light', 'regular', 'bold']
plt.figure(figsize=(12, 4))
for i, lc in enumerate(['left', 'center', 'right']):
    plt.subplot(1, 3, i+1)
    plt.title(label=lc, loc=lc, fontsize=12+i*5, fontweight=weight[i],
pad=10+i*15)
plt.show()

# 28
import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [i*2 for i in range(10)]

plt.plot(x, y)
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.show()

# 29
import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [i*2 for i in range(10)]

plt.plot(x, y)
plt.xlabel('Ось X\nНезависимая величина', fontsize=14, fontweight='bold')
plt.ylabel('Ось Y\nЗависимая величина', fontsize=14, fontweight='bold')
plt.show()

# 30
import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [i*2 for i in range(10)]

plt.text(0, 7, 'HELLO!', fontsize=15)
plt.plot(range(0,10), range(0,10))
plt.show()

# 31
import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [i*2 for i in range(10)]

bbox_properties=dict(boxstyle='darrow, pad=0.3', ec='k', fc='y', ls='-', lw=3)

plt.text(2, 7, 'HELLO!', fontsize=15, bbox=bbox_properties)
plt.plot(range(0,10), range(0,10))
plt.show()

# 32
import math
import matplotlib.pyplot as plt

x = list(range(-5, 6))
y = [i**2 for i in x]

plt.annotate('min', xy=(0, 0), xycoords='data', xytext=(0, 10), textcoords='data', arrowprops=dict(facecolor='g'))
plt.plot(x, y)
plt.show()

# 33
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))

arrows = ['-', '->', '-[', '|-|', '-|>', '<-', '<->', '<|-', '<|-|>', 'fancy', 'simple', 'wedge']

bbox_properties=dict(
    boxstyle='round,pad=0.2',
    ec='k',
    fc='w',
    ls='-',
    lw=1
)
ofs_x = 0
ofs_y = 0

for i, ar in enumerate(arrows):
    if i == 6: ofs_x = 0.5

    plt.annotate(ar, xy=(0.4+ofs_x, 0.92-ofs_y), xycoords='data',
        xytext=(0.05+ofs_x, 0.9-ofs_y), textcoords='data', fontsize=17,
        bbox=bbox_properties,
        arrowprops=dict(arrowstyle=ar)
    )
    if ofs_y == 0.75: ofs_y = 0
    else: ofs_y += 0.15
plt.show()

# 34
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 3, figsize=(12, 7))
conn_style=[
    'angle,angleA=90,angleB=0,rad=0.0',
    'angle3,angleA=90,angleB=0',
    'arc,angleA=0,angleB=0,armA=0,armB=40,rad=0.0',
    'arc3,rad=-1.0',
    'bar,armA=0.0,armB=0.0,fraction=0.1,angle=70',
    'bar,fraction=-0.5,angle=180',
]
for i in range(2):
    for j in range(3):
        axs[i, j].text(0.1, 0.5, '\n'.join(conn_style[i*3+j].split(',')))
        axs[i, j].annotate('text', xy=(0.2, 0.2), xycoords='data',
            xytext=(0.7, 0.8), textcoords='data',
            arrowprops=dict(arrowstyle='->',
connectionstyle=conn_style[i*3+j]))
plt.show()

# 35 
import matplotlib.pyplot as plt

plt.title('Title', alpha=0.5, color='r', fontsize=18, fontstyle='italic', fontweight='bold', linespacing=10)
plt.plot(range(0,10), range(0,10))
plt.show()

# 36
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.title('Title', fontproperties=FontProperties(family='monospace', style='italic', weight='heavy', size=15))
plt.plot(range(0,10), range(0,10))
plt.show()

# 37
import matplotlib.pyplot as plt

plt.title('Title', fontsize=17, position=(0.7, 0.2), rotation='vertical')
plt.plot(range(0,10), range(0,10))
plt.show()

# 38
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

bbox_properties=dict(
    boxstyle='rarrow, pad=0.3',
    ec='g',
    fc='r',
    ls='-',
    lw=3
)
plt.title('Title', fontsize=17, bbox=bbox_properties, position=(0.5, 0.85))
plt.plot(range(0,10), range(0,10))
plt.show()

# 39
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

vals = np.random.randint(10, size=(7, 7))

plt.pcolor(vals)
plt.show()

# 40
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

vals = np.random.randint(10, size=(7, 7))

plt.pcolor(vals)
plt.colorbar()
plt.show()

# 41
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

vals = np.random.randint(10, size=(7, 7))

plt.pcolor(vals, cmap=plt.get_cmap('viridis', 11) )
plt.colorbar()
plt.show()

# 42
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

np.random.seed(123)

vals = np.random.randint(11, size=(7, 7))
fig, ax = plt.subplots()
gr = ax.pcolor(vals)
axins = inset_axes(ax, width="7%", height="50%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)

plt.colorbar(gr, cax=axins)
plt.show()

# 43 
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

np.random.seed(123)

vals = np.random.randint(11, size=(7, 7))
fig, ax = plt.subplots()
gr = ax.pcolor(vals)
axins = inset_axes(ax, width="7%", height="50%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)

plt.colorbar(gr, cax=axins, ticks=[0, 5, 10], label='Value')
plt.show()

# 44
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

np.random.seed(123)

vals = np.random.randint(11, size=(7, 7))
fig, ax = plt.subplots()
gr = ax.pcolor(vals)
axins = inset_axes(ax, width="7%", height="50%", loc='lower left', bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
cbar = plt.colorbar(gr, cax=axins, ticks=[0, 5, 10], label='Value')
cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])

plt.show()

# 45
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123)

vals = np.random.randint(10, size=(7, 7))

plt.pcolor(vals, cmap=plt.get_cmap('viridis', 11))
plt.colorbar(orientation='horizontal',
    shrink=0.9, extend='max', extendfrac=0.2,
    extendrect=False, drawedges=False)
plt.show()