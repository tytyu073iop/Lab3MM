import math
import random
import numpy as np
from graphs import *
from scipy.stats import norm, lognorm, cauchy, kstest, chisquare
from scipy import stats


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

def generate_lognormal_inverse(n, mu=0, sigma=1, random_state=None):
    """
    Генерирует выборку из логнормального распределения методом обратной функции.
    
    Параметры:
    -----------
    n : int
        Размер выборки
    mu : float
        Среднее соответствующего нормального распределения (ln(X) ~ N(mu, sigma^2))
    sigma : float
        Стандартное отклонение соответствующего нормального распределения (>0)
    random_state : int или None
        Seed для воспроизводимости
        
    Возвращает:
    -----------
    sample : numpy.ndarray
        Массив размером n из логнормального распределения
    """
    if sigma <= 0:
        raise ValueError("Параметр sigma должен быть положительным")
    
    # Инициализация генератора случайных чисел
    rng = np.random.RandomState(random_state)
    
    # 1. Генерируем равномерное распределение на [0, 1]
    uniform_samples = rng.random(n)
    
    # 2. Применяем обратную функцию нормального распределения
    # (метод обратной функции для нормального распределения)
    normal_samples = norm.ppf(uniform_samples)
    
    # 3. Преобразуем нормальные выборки к нужным параметрам
    normal_samples_transformed = mu + sigma * normal_samples
    
    # 4. Экспоненцируем, чтобы получить логнормальное распределение
    lognormal_samples = np.exp(normal_samples_transformed)
    
    return lognormal_samples

def calculate_lognormal_parameters(mu_normal, sigma_normal):
    """
    Вычисляет параметры логнормального распределения
    на основе параметров соответствующего нормального распределения.
    """
    # Среднее логнормального распределения
    mean_lognormal = np.exp(mu_normal + sigma_normal**2 / 2)
    
    # Дисперсия логнормального распределения
    variance_lognormal = (np.exp(sigma_normal**2) - 1) * np.exp(2 * mu_normal + sigma_normal**2)
    
    # Медиана логнормального распределения
    median_lognormal = np.exp(mu_normal)
    
    # Мода логнормального распределения
    mode_lognormal = np.exp(mu_normal - sigma_normal**2)
    
    return {
        'mean': mean_lognormal,
        'variance': variance_lognormal,
        'median': median_lognormal,
        'mode': mode_lognormal
    }

def Koshi(n, a, b):
    u = np.random.uniform(0,1, n)

    sample = a+b*np.tan(np.pi*(u-0.5))

    return sample


def run_kolmogorov_smirnov_test(dist_name, arr, params, alpha=0.05):
    """Выполняет критерий Колмогорова-Смирнова для одной выборки"""
    # Генерируем выборку из заданного распределения
    if dist_name == 'normal':
        # Тестируем против нормального распределения
        result = kstest(arr, 'norm', args=(params['loc'], params['scale']))
    elif dist_name == 'lognormal':
        # Тестируем против логнормального распределения
        # Для KS теста нужно передать параметры правильно
        result = kstest(arr, 'lognorm', args=(params['s'], 0, params['scale']))
    elif dist_name == 'cauchy':
        # Тестируем против распределения Коши
        result = kstest(arr, 'cauchy', args=(params['loc'], params['scale']))
    
    return result.pvalue < alpha  # Отвергаем H0 если p-value < alpha

def run_chi_square_test(dist_name, arr, params, alpha=0.05, n_bins=10):
    """Выполняет критерий хи-квадрат Пирсона для одной выборки"""
    # Генерируем выборку
    if dist_name == 'normal':
        # Создаем бины на основе квантилей нормального распределения
        bins = np.linspace(norm.ppf(0.001, **params), 
                          norm.ppf(0.999, **params), 
                          n_bins + 1)
    elif dist_name == 'lognormal':
        bins = np.linspace(lognorm.ppf(0.001, **params), 
                          lognorm.ppf(0.999, **params), 
                          n_bins + 1)
    elif dist_name == 'cauchy':
        # Для Коши используем более широкие границы из-за тяжелых хвостов
        bins = np.linspace(cauchy.ppf(0.01, **params), 
                          cauchy.ppf(0.99, **params), 
                          n_bins + 1)
    
    # Вычисляем наблюдаемые частоты
    observed, _ = np.histogram(arr, bins=bins)
    
    # Вычисляем ожидаемые частоты
    if dist_name == 'normal':
        cdf_values = norm.cdf(bins, **params)
    elif dist_name == 'lognormal':
        # Для логнормального распределения в scipy параметры передаются иначе
        cdf_values = lognorm.cdf(bins, **params)
    elif dist_name == 'cauchy':
        cdf_values = cauchy.cdf(bins, **params)
    
    expected = len(arr) * np.diff(cdf_values)
    
    # Избегаем нулевых ожидаемых частот (объединяем соседние бины если нужно)
    # Для простоты просто пропустим тест если есть проблемы
    if np.any(expected < 5):
        # Объединяем соседние бины с малыми ожидаемыми частотами
        valid_indices = expected >= 1
        observed = observed[valid_indices]
        expected = expected[valid_indices]
        
        if len(observed) < 2 or np.sum(expected) < 5:
            return False
    
    # Выполняем тест хи-квадрат
    chi2_stat = np.sum((observed - expected)**2 / expected)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=len(observed) - 1 - len(params))
    
    return p_value < alpha

def estimate_type1_error(dist_name, Aarr, params, test_func):
    """Оценивает вероятность ошибки I рода для заданного теста"""
    rejections = 0
    
    for i in Aarr:
        if test_func(dist_name, i, params):
            rejections += 1
    
    return rejections / len(Aarr)


if __name__ == "__main__":
    m = -3
    s = 4
    n = 1000
    mArr = Muller(n, m, s)
    print("Muller: ", mArr, "Kolmogorov pass: ", not run_kolmogorov_smirnov_test('normal', mArr, {'loc': 0, 'scale': 3}))
    print("Error: ", estimate_type1_error('normal', [Muller(n, m, s)] * 100, {'loc': 0, 'scale': 3}, run_kolmogorov_smirnov_test))
    print("Pirson: ", not run_chi_square_test('normal', mArr, {'loc': 0, 'scale': 3}))
    print("Error: ", estimate_type1_error('normal', [Muller(n, m, s)] * 100, {'loc': 0, 'scale': 3}, run_chi_square_test))
    plot_value_occurrences(mArr, "Normal distribution", 10)
    print("diff between expected nesmeszonnoe and istinaya: ", abs(np.mean(mArr) - m))
    print("diff between dispersiya nesmeszonnoe and istinaya: ", abs(np.var(mArr, ddof=1) - pow(s,2)))

    lnArr = generate_lognormal_inverse(1000, 0, 2)
    jarr = calculate_lognormal_parameters(0, 2)
    print("LogNormal: ", lnArr, "kolmog pass: ", not run_kolmogorov_smirnov_test('lognormal', lnArr, {'s': 2, 'scale': np.exp(0)}))
    print("Error: ", estimate_type1_error('lognormal', [generate_lognormal_inverse(1000, 0, 2)] * 100, {'s': 2, 'scale': np.exp(0)}, run_kolmogorov_smirnov_test))
    print("Pirson: ", not run_chi_square_test('lognormal', lnArr, {'s': 2, 'scale': np.exp(0)}))
    print("Error: ", estimate_type1_error('lognormal', [generate_lognormal_inverse(1000, 0, 2)] * 100, {'s': 2, 'scale': np.exp(0)}, run_kolmogorov_smirnov_test))
    plot_value_occurrences(lnArr, "LogNormal distribution", 10)
    mean_relative_error = abs(np.mean(lnArr) - jarr['mean']) / jarr['mean']
    var_relative_error = abs(np.var(lnArr, ddof=1) - jarr['variance']) / jarr['variance']
    print(f"Relative mean error: {mean_relative_error:.2%}")
    print(f"Relative variance error: {var_relative_error:.2%}")

    KArr = Koshi(n, 1,2)
    print("Koshi: ", KArr, "kolmogorov pass: ", not run_kolmogorov_smirnov_test('cauchy', KArr, {'loc': 1, 'scale': 2}))
    print("Error: ", estimate_type1_error('cauchy', [Koshi(n, 1,2)] * 100, {'loc': 1, 'scale': 2}, run_kolmogorov_smirnov_test))
    print("Pirson: ", not run_chi_square_test('cauchy', KArr, {'loc': 1, 'scale': 2}))
    print("Error: ", estimate_type1_error('cauchy', [Koshi(n, 1,2)] * 100, {'loc': 1, 'scale': 2}, run_chi_square_test))
    plot_value_occurrences(KArr, "Koshi distribution", 10)
    # no mean or distribution

    