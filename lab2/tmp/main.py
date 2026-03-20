import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

def create_vector():
    """
    Создает массив от 0 до 9.
    
    Returns:
        numpy.ndarray: Массив чисел от 0 до 9 включительно
    """
    return np.arange(10)

def create_matrix():
    """
    Создает матрицу 5x5 со случайными числами [0,1].
    
    Returns:
        numpy.ndarray: Матрица 5x5 со случайными значениями от 0 до 1
    """
    return np.random.rand(5, 5)

def reshape_vector(vec):
    """
    Преобразовывает (10,) -> (2,5)

    Args:
        vec (numpy.ndarray): Входной массив формы (10,)
    
    Returns:
        numpy.ndarray: Преобразованный массив формы (2, 5)
    """
    return vec.reshape(2, 5)

def transpose_matrix(mat):
    """
    Транспонирование матрицы.
    
    Args:
        mat (numpy.ndarray): Входная матрица
    
    Returns:
        numpy.ndarray: Транспонированная матрица
    """
    return np.transpose(mat)

def vector_add(a, b):
    """
    Сложение векторов одинаковой длины.
    (Векторизация без циклов)
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        numpy.ndarray: Результат поэлементного сложения
    """
    return a + b

def scalar_multiply(vec, scalar):
    """
    Умножение вектора на число.
    
    Args:
        vec (numpy.ndarray): Входной вектор
        scalar (float/int): Число для умножения
    
    Returns:
        numpy.ndarray: Результат умножения вектора на скаляр
    """
    return vec * scalar

def elementwise_multiply(a, b):
    """
    Поэлементное умножение.
    
    Args:
        a (numpy.ndarray): Первый вектор/матрица
        b (numpy.ndarray): Второй вектор/матрица
    
    Returns:
        numpy.ndarray: Результат поэлементного умножения
    """
    return a * b

def dot_product(a, b):
    """
    Скалярное произведение.
    
    Args:
        a (numpy.ndarray): Первый вектор
        b (numpy.ndarray): Второй вектор
    
    Returns:
        float: Скалярное произведение векторов
    """
    return np.dot(a, b)

def matrix_multiply(a, b):
    """
    Умножение матриц.
    
    Args:
        a (numpy.ndarray): Первая матрица
        b (numpy.ndarray): Вторая матрица
    
    Returns:
        numpy.ndarray: Результат умножения матриц
    """
    return a @ b

def matrix_determinant(a):
    """
    Определитель матрицы.
    
    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        float: Определитель матрицы
    """
    return np.linalg.det(a)

def matrix_inverse(a):
    """
    Обратная матрица.

    Args:
        a (numpy.ndarray): Квадратная матрица
    
    Returns:
        numpy.ndarray: Обратная матрица
    """
    return np.linalg.inv(a)

def solve_linear_system(a, b):
    """
    Решает систему Ax = b
    
    Args:
        a (numpy.ndarray): Матрица коэффициентов A
        b (numpy.ndarray): Вектор свободных членов b
    
    Returns:
        numpy.ndarray: Решение системы x
    """
    return np.linalg.solve(a, b)

def load_dataset(path="data/students_scores.csv"):
    """
    Загружает CSV и вернуть NumPy массив.
    
    Args:
        path (str): Путь к CSV файлу
    
    Returns:
        numpy.ndarray: Загруженные данные в виде массива
    """
    return pd.read_csv(path).to_numpy()

def statistical_analysis(data):
    """
    Данные — это результаты экзамена по математике.
    Функция оценивает:
    - средний балл
    - медиану
    - стандартное отклонение
    - минимум
    - максимум
    - 25 и 75 перцентили
    
    Args:
        data (numpy.ndarray): Одномерный массив данных
    
    Returns:
        dict: Словарь со статистическими показателями
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'percentile_25': np.percentile(data, 25),
        'percentile_75': np.percentile(data, 75)
    }

def normalize_data(data):
    """
    Min-Max нормализация.
    
    Формула: (x - min) / (max - min)
    
    Args:
        data (numpy.ndarray): Входной массив данных
    
    Returns:
        numpy.ndarray: Нормализованный массив данных в диапазоне [0, 1]
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def plot_histogram(data):
    """
    Строит гистограмму распределения оценок по математике.
    
    Args:
        data (numpy.ndarray): Данные для гистограммы
    """
    plt.hist(data, bins=10, edgecolor='black', color='skyblue')
    plt.title('Распределение оценок')
    plt.xlabel('Оценки')
    plt.ylabel('Количество студентов')
    plt.savefig('plots/histogram.png')
    plt.close()

def plot_heatmap(matrix):
    """
    Строит тепловую карту корреляции предметов.
    
    Args:
        matrix (numpy.ndarray): Матрица корреляции
    """
    sns.heatmap(matrix, annot=True, cmap='coolwarm')
    plt.title('Тепловая карта корреляции предметов')
    plt.savefig('plots/heatmap.png')
    plt.close()

def plot_line(x, y):
    """
    Строит график зависимости: студент -> оценка по математике.
    
    Args:
        x (numpy.ndarray): Номера студентов
        y (numpy.ndarray): Оценки студентов
    """
    plt.plot(x, y, marker='o')
    plt.title('Зависимость: студент - оценка по математике')
    plt.xlabel('Студент')
    plt.ylabel('Оценка')
    plt.savefig('plots/line_plot.png')
    plt.close()

if __name__ == "__main__":
    print("Запустите python3 -m pytest test.py -v для проверки лабораторной работы.")