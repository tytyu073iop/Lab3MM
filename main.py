import math
import random
import numpy as np
from graphs import *

def Muller(n, m=0, s=1):
    """
    Генерация нормально распределенных случайных величин 
    с использованием преобразования Мюллера
    """
    results = []
    
    for i in range(int(n/2)+1):
        # Генерируем две равномерно распределенные случайные величины
        u1 = random.random()
        u2 = random.random()
        
        # Преобразование Бокса-Мюллера
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        
        # Масштабируем и сдвигаем
        x0 = m + s * z0
        x1 = m + s * z1
        
        results.extend([x0, x1])
    
    return results[:n]  # Возвращаем ровно n значений


if __name__ == "__main__":
    m = 0
    s = 3
    n = 1000
    mArr = Muller(n, m, s)
    print("Muller: ", mArr)
    plot_value_occurrences(mArr, "Normal distribution", 10)
    print("diff between expected nesmeszonnoe and istinaya: ", abs(np.mean(mArr) - m))
    print("diff between dispersiya nesmeszonnoe and istinaya: ", abs(np.var(mArr, ddof=1) - pow(s,2)))

    