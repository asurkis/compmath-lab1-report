import numpy as np


# Поиск подходящей перестановки методом коллапса волновой функции
def wave_function_collapse(non_strict: np.ndarray,
                           strict: np.ndarray,
                           permutation: [int] = []) -> [int]:
    size = len(non_strict)
    position = len(permutation)
    if position == size:
        for i in range(size):
            if strict[i][permutation[i]]:
                return permutation
        return None
    for i in range(size):
        if non_strict[i][position] and i not in permutation:
            res = wave_function_collapse(non_strict, strict, permutation + [i])
            if res is not None:
                return res
    return None


def solve_prepared(c_matrix: np.ndarray,
                   d_vector: np.ndarray,
                   precision: float) -> np.ndarray:
    x_vector = np.zeros((x_count,))
    diff_vector: np.ndarray
    iteration_count = 0
    while True:
        iteration_count += 1
        x_vector_new: np.ndarray = (c_matrix @ x_vector) + d_vector
        diff_vector = abs(x_vector_new - x_vector)
        x_vector = x_vector_new
        if (diff_vector < precision).all():
            break
    print('Количество итераций:', iteration_count)
    print('|x_i^{(k)} - x_i^{(k-1)}| =', diff_vector)
    return x_vector


def solve_equation(ab_list: [[float]],
                   x_count: int,
                   eq_count: int,
                   precision: float) -> np.ndarray:
    # Матрица A|B
    a_matrix_original: np.ndarray = np.array(
        [[ab_list[i][j] if i < eq_count else 0
          for j in range(x_count)]
         for i in range(x_count)])
    b_vector_original = np.array([ab_list[i][-1] if i < eq_count else 0
                         for i in range(x_count)])

    # Проверка диагонального преобладания
    print('Проверка диагонального преобладания: ')
    # Подготовка матриц для перебора
    sums: np.ndarray = abs(a_matrix_original).sum(axis=1)
    double_abs: np.ndarray = 2 * abs(a_matrix_original)
    collapse_matrix_non_strict: np.ndarray = double_abs >= sums
    collapse_matrix_strict: np.ndarray = double_abs > sums

    a_matrix = a_matrix_original
    b_vector = b_vector_original

    if collapse_matrix_non_strict.diagonal().all() \
            and collapse_matrix_strict.diagonal().any():
        print('Диагональное преобладание изначально')
    else:
        print('Диагонального преобладания изначально нет')
        permutation = wave_function_collapse(
            collapse_matrix_non_strict,
            collapse_matrix_strict)
        if permutation is None:
            print('Диагональное преобладание невозможно')
        else:
            print('Диагональное преобладание с перестановкой', permutation)
            a_matrix = np.array([a_matrix_original[p] for p in permutation])
            b_vector = np.array([b_vector_original[p] for p in permutation])

    c_matrix = np.eye(x_count) - a_matrix / a_matrix.diagonal()
    d_vector = b_vector / a_matrix.diagonal()
    x_vector = solve_prepared(c_matrix, d_vector, precision)
    print('A @ X - B =', (a_matrix_original @ x_vector) - b_vector_original)
    return x_vector


# Ввод данных
x_count = int(input('Введите количество переменных: '))
print(x_count)
eq_count = int(input('Введите количество уравнений: '))
print(eq_count)
precision = float(input('Введите требуемую точность: '))
print(precision)
print()
print('Введите матрицу системы: ')
ab_list = [[float(s) for s in input().split(' ') if s != '']
           for _ in range(eq_count)]

x_result = solve_equation(ab_list, x_count, eq_count, precision)
print('X =', x_result)
