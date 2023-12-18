# Лаборатортная работа
## Сложность: Rare
## Вариант №2
### Задание
1. Создайте в каталоге для данной ЛР в своём репозитории виртуальное окружение и установите в него ```matplotlib``` и ```numpy```. Создайте файл ```requirements.txt```.
2. Откройте книгу и выполните уроки 1-3. Первый урок можно начинать со стр. 8.
3. Выберите одну из неразрывных функции своего варианта из лабораторной работы №2, постройте график этой функции и касательную к ней. Добавьте на график заголовок, подписи осей, легенду, сетку, а также аннотацию к точке касания.
4. Добавьте в корень своего репозитория файл ```.gitignore```, перед тем как делать очередной коммит.
5. Оформите отчёт в ```README.md```. Отчёт должен содержать: 
    - графики, построенные во время выполнения уроков из книги 
    - объяснения процесса решения и график по заданию 4
6. Склонируйте этот репозиторий НЕ в ваш репозиторий, а рядом. Изучите использование этого инструмента и создайте pdf-версию своего отчёта из ```README.md```. Добавьте её в репозиторий.
### Ход работы
#### Задание 2
#### 1
![Alt text](Figure_1.png)
#### 2
![Alt text](Figure_2.png)
#### 3
![Alt text](Figure_3.png)
#### 4
![Alt text](Figure_4.png)
#### 5
![Alt text](Figure_5.png)
#### 6
![Alt text](Figure_6.png)
#### 7
![Alt text](Figure_7.png)
#### 8
![Alt text](Figure_8.png)
#### 9
![Alt text](Figure_9.png)
#### 10
![Alt text](Figure_10.png)
#### 11
![Alt text](Figure_11.png)
#### 12
![Alt text](Figure_12.png)
#### 13
![Alt text](Figure_13.png)
#### 14
![Alt text](Figure_14.png)
#### 15
![Alt text](Figure_15.png)
#### 16
![Alt text](Figure_16.png)
#### 17
![Alt text](Figure_17.png)
#### 18
![Alt text](Figure_18.png)
#### 19
![Alt text](Figure_19.png)
#### 20
![Alt text](Figure_20.png)
#### 21
![Alt text](Figure_21.png)
#### 22
![Alt text](Figure_22.png)
#### 23
![Alt text](Figure_23.png)
#### 24
![Alt text](Figure_24.png)
#### 25
![Alt text](Figure_25.png)
#### 26
![Alt text](Figure_26.png)
#### 27
![Alt text](Figure_27.png)
#### 28
![Alt text](Figure_28.png)
#### 29
![Alt text](Figure_29.png)
#### 30
![Alt text](Figure_30.png)
#### 31
![Alt text](Figure_31.png)
#### 32
![Alt text](Figure_32.png)
#### 33
![Alt text](Figure_33.png)
#### 34
![Alt text](Figure_34.png)
#### 35
![Alt text](Figure_35.png)
#### 36
![Alt text](Figure_36.png)
#### 37
![Alt text](Figure_37.png)
#### 38
![Alt text](Figure_38.png)
#### 39
![Alt text](Figure_39.png)
#### 40
![Alt text](Figure_40.png)
#### 41
![Alt text](Figure_41.png)
#### 42
![Alt text](Figure_42.png)
#### 43
![Alt text](Figure_43.png)
#### 44
![Alt text](Figure_44.png)
#### 45
![Alt text](Figure_45.png)
#### Задание 3

$f(x) = e^{sin{x}}$

```python
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
```
#### Иллюстрация решения задачи
![Alt text](Figure_46.png)

