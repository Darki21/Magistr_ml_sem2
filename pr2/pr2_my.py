import numpy as np
import pandas as pd
# Найти уравнение вида: y = b₀ + b₁x₁ + b₂x₂
# Где y - стоимость авто, x₁ - возраст, x₂ - пробег

y = np.array([55, 80, 55, 83, 86, 107, 85, 50, 75, 50, 95, 40, 100, 40, 60])
x1 = np.array([14, 11, 14, 8, 8, 7, 7, 14, 7, 17, 7, 14, 6, 18, 12])
x2 = np.array([100, 150, 195, 118, 77, 100, 100, 180, 79, 90, 100, 184, 76, 83, 100])

n = len(y)
# ========== 0. ВЫЧИСЛЕНИЕ промедуточных значений ==========
x1y = x1*y
x2y = x2*y
x1x2 = x1*x2
x1_square = x1*x1
x2_square = x2*x2
y_square = y*y
print(f"\nПромедуточные значения:")
data = np.array([y,x1,x2,x1y,x2y,x1x2,x1_square,x2_square,y_square])
print(data.T)
print("="*70)
print("ЧАСТЬ 1: РУЧНОЙ ПОДСЧЕТ ")
print("="*70)
# ========== 1. ВЫЧИСЛЕНИЕ СРЕДНИХ ЗНАЧЕНИЙ ==========
y_mean = sum(y) / n
x1_mean = sum(x1) / n
x2_mean = sum(x2) / n

print(f"\nСредние значение:")
print("ȳ = ", y_mean)
print("x̄₁ = ", x1_mean)
print("x̄₂ = ", x2_mean)

# ========== 2. ВЫЧИСЛЕНИЕ СУММ КВАДРАТОВ И ПРОИЗВЕДЕНИЙ ==========

sum_x1 = sum(x1)
sum_x2 = sum(x2)
sum_y = sum(y)

sum_x1_sq = sum(x1_square)
sum_x2_sq = sum(x2_square)
sum_y_sq = sum(y_square)

sum_x1_x2 = sum(x1x2)
sum_x1_y = sum(x1y)
sum_x2_y = sum(x2y)

print(f"\nНеобходимые суммы:")
print(f"Σx₁ = {sum_x1}")
print(f"Σx₂ = {sum_x2}")
print(f"Σy = {sum_y}")
print(f"Σx₁² = {sum_x1_sq}")
print(f"Σx₂² = {sum_x2_sq}")
print(f"Σy² = {sum_y_sq}")
print(f"Σx₁x₂ = {sum_x1_x2}")
print(f"Σx₁y = {sum_x1_y}")
print(f"Σx₂y = {sum_x2_y}")

# ========== 3. МАТРИЦА СИСТЕМЫ НОРМАЛЬНЫХ УРАВНЕНИЙ ==========
# Система уравнений:
# b0*n + b1*Σx1 + b2*Σx2 = Σy
# b0*Σx1 + b1*Σx1² + b2*Σx1x2 = Σx1y
# b0*Σx2 + b1*Σx1x2 + b2*Σx2² = Σx2y

# Матрица коэффициентов (без свободных членов)
A = np.array([
    [n, sum_x1, sum_x2],
    [sum_x1, sum_x1_sq, sum_x1_x2],
    [sum_x2, sum_x1_x2, sum_x2_sq]
])/n

# Вектор правых частей
B = np.array([sum_y, sum_x1_y, sum_x2_y]) / n

print(f"\nМатрица системы нормальных уравнений:")
print(A)
print(f"\nВектор правых частей: {B}")

# ========== 4. ПРОВЕРКА ОПРЕДЕЛИТЕЛЯ ==========
det_A = np.linalg.det(A)
print(f"\nОпределитель матрицы: det = {det_A}")

if abs(det_A) < 1e-10:
    print("ВНИМАНИЕ: Определитель близок к нулю! Существует коллинеарность.")
else:
    print("Определитель далек от нуля - решение существует.")

# ========== 5. РЕШЕНИЕ СИСТЕМЫ (метод Гаусса вручную) ==========
# Расширенная матрица Гаусса
gauss_matrix = np.column_stack([A, B])
print(f"\nРасширенная матрица Гаусса:")
print(gauss_matrix)
try:
    b_manual = np.linalg.solve(A, B)
except:
    print("Нет решения. Коллинеарность среди Х?")

print(f"\nКоэффициенты регрессии (ручной подсчет):")
print(f"b₀ = { b_manual[0]}")
print(f"b₁ = { b_manual[1]}")
print(f"b₂ = { b_manual[2]}")


# ========== 6. СРЕДНЕЕ КВАДРАТИЧЕСКОЕ ОТКЛОНЕНИЕ ==========
def std_dev_manual(x, mean):
    variance = sum([(xi - mean)**2 for xi in x]) / n
    return np.sqrt(variance)

y_std = std_dev_manual(y, y_mean)
x1_std = std_dev_manual(x1, x1_mean)
x2_std = std_dev_manual(x2, x2_mean)

print(f"\nСредние квадратические отклонения:")
print(f"σy = {y_std}")
print(f"σx₁ = {x1_std}")
print(f"σx₂ = {x2_std}")

# ========== 7. КОЭФФИЦИЕНТЫ ПАРНОЙ КОРРЕЛЯЦИИ (Пирсон) ==========
def pearson_corr_manual(y_mean, x_mean, xy_mean, y_std, x_std):
    covariance = (xy_mean-y_mean*x_mean)
    return covariance/(y_std*x_std)

r_yx1_manual = pearson_corr_manual(y_mean, x1_mean, sum_x1_y/n, y_std, x1_std)
r_yx2_manual = pearson_corr_manual(y_mean, x2_mean, sum_x2_y/n, y_std, x2_std)
r_x1x2_manual = pearson_corr_manual(x1_mean, x2_mean, sum_x1_x2/n, x1_std, x2_std)

print(f"\nКоэффициенты парной корреляции (ручной подсчет):")
print(f"r(y, x₁) = {r_yx1_manual}")
print(f"r(y, x₂) = {r_yx2_manual}")
print(f"r(x₁, x₂) = {r_x1x2_manual}")

# ========== 8. КОЭФФИЦИЕНТЫ ЧАСТНОЙ КОРРЕЛЯЦИИ ==========
# r(y,x1|x2) = (r_yx1 - r_yx2*r_x1x2) / sqrt((1-r_yx2²)(1-r_x1x2²))
# r(y,x2|x1) = (r_yx2 - r_yx1*r_x1x2) / sqrt((1-r_yx1²)(1-r_x1x2²))

r_yx1_x2_manual = (r_yx1_manual - r_yx2_manual * r_x1x2_manual) / np.sqrt((1 - r_yx2_manual**2) * (1 - r_x1x2_manual**2))

r_yx2_x1_manual = (r_yx2_manual - r_yx1_manual * r_x1x2_manual) / np.sqrt((1 - r_yx1_manual**2) * (1 - r_x1x2_manual**2))

print(f"\nКоэффициенты частной корреляции (ручной подсчет):")
print(f"r(y, x₁|x₂) = {r_yx1_x2_manual}")
print(f"r(y, x₂|x₁) = {r_yx2_x1_manual}")

# ========== 9. КОЭФФИЦИЕНТ МНОЖЕСТВЕННОЙ КОРРЕЛЯЦИИ ==========
R_manual = np.sqrt((r_yx1_manual**2 + r_yx2_manual**2 - 2*r_yx1_manual*r_yx2_manual*r_x1x2_manual) / 
                   (1 - r_x1x2_manual**2))

R_squared_manual = R_manual**2

print(f"\nКоэффициент множественной корреляции (ручной подсчет):")
print(f"R = {R_manual}")
print(f"R² = {R_squared_manual}")
print(f" дисперсия в процентах: {R_squared_manual*100}%")

def cheddock_scale(r):
    abs_r = abs(r)
    if abs_r < 0.3:
        return "очень слабая"
    elif abs_r < 0.5:
        return "слабая"
    elif abs_r < 0.7:
        return "заметная"
    elif abs_r < 0.9:
        return "тесная"
    else:
        return "весьма тесная"
    
print(f"\n   ИНТЕРПРЕТАЦИЯ (шкала Чеддока):")
print(f"   • Связь цены с возрастом (без учета пробега):")
print(f"     {cheddock_scale(r_yx1_x2_manual)} ({'прямая' if r_yx1_x2_manual > 0 else 'обратная'})")
print(f"   • Связь цены с пробегом (без учета возраста):")
print(f"     {cheddock_scale(r_yx2_x1_manual)} ({'прямая' if r_yx2_x1_manual > 0 else 'обратная'})")


print("\n" + "="*70)
print("ЧАСТЬ 2: ПОДСЧЕТ С ПОМОЩЬЮ NUMPY")
print("="*70)
# ========== 1. РЕГРЕССИЯ ЧЕРЕZ NUMPY ==========
X = np.column_stack([np.ones(n), x1, x2])
b_numpy = np.linalg.inv(X.T @ X) @ X.T @ y

print(f"\nКоэффициенты регрессии (NumPy):")
print(f"b₀ = {b_numpy[0]}")
print(f"b₁ = {b_numpy[1]}")
print(f"b₂ = {b_numpy[2]}")

# ========== 2. КОРРЕЛЯЦИИ ЧЕРЕЗ NUMPY ==========
r_yx1_numpy = np.corrcoef(y, x1)[0, 1]
r_yx2_numpy = np.corrcoef(y, x2)[0, 1]
r_x1x2_numpy = np.corrcoef(x1, x2)[0, 1]

print(f"\nКоэффициенты парной корреляции (NumPy):")
print(f"r(y, x₁) = {r_yx1_numpy}")
print(f"r(y, x₂) = {r_yx2_numpy}")
print(f"r(x₁, x₂) = {r_x1x2_numpy}")

# ========== 3. ЧАСТНЫЕ КОРРЕЛЯЦИИ (формулы те же) ==========
r_yx1_x2_numpy = (r_yx1_numpy - r_yx2_numpy * r_x1x2_numpy) / \
                 np.sqrt((1 - r_yx2_numpy**2) * (1 - r_x1x2_numpy**2))

r_yx2_x1_numpy = (r_yx2_numpy - r_yx1_numpy * r_x1x2_numpy) / \
                 np.sqrt((1 - r_yx1_numpy**2) * (1 - r_x1x2_numpy**2))

print(f"\nКоэффициенты частной корреляции (NumPy):")
print(f"r(y, x₁|x₂) = {r_yx1_x2_numpy}")
print(f"r(y, x₂|x₁) = {r_yx2_x1_numpy}")

# ========== 4. МНОЖЕСТВЕННАЯ КОРРЕЛЯЦИЯ ==========
R_numpy = np.sqrt((r_yx1_numpy**2 + r_yx2_numpy**2 - 2*r_yx1_numpy*r_yx2_numpy*r_x1x2_numpy) / 
                 (1 - r_x1x2_numpy**2))
R_squared_numpy = R_numpy**2

print(f"\nКоэффициент множественной корреляции (NumPy):")
print(f"R = {R_numpy}")
print(f"R² = {R_squared_numpy}")

print("\n" + "="*70)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*70)

print(f"\nКоэффициенты регрессии:")
print(f"b₀: ручной={b_manual[0]}, NumPy={b_numpy[0]}, разница={abs(b_manual[0]-b_numpy[0]):.2e}")
print(f"b₁: ручной={b_manual[1]}, NumPy={b_numpy[1]}, разница={abs(b_manual[1]-b_numpy[1]):.2e}")
print(f"b₂: ручной={b_manual[2]}, NumPy={b_numpy[2]}, разница={abs(b_manual[2]-b_numpy[2]):.2e}")

print(f"\nКоэффициенты корреляции:")
print(f"r(y,x₁): ручной={r_yx1_manual}, NumPy={r_yx1_numpy}, разница={abs(r_yx1_manual-r_yx1_numpy):.2e}")
print(f"r(y,x₂): ручной={r_yx2_manual}, NumPy={r_yx2_numpy}, разница={abs(r_yx2_manual-r_yx2_numpy):.2e}")
print(f"R²: ручной={R_squared_manual}, NumPy={R_squared_numpy}, разница={abs(R_squared_manual-R_squared_numpy):.2e}")

print("\n" + "="*70)
print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
print("="*70)

print(f"\nУравнение регрессии:")
print(f"y = {b_numpy[0]} + ({b_numpy[1]})·x₁ + ({b_numpy[2]})·x₂")
# Интерпретация по шкале Чеддока
print(f"\n   ИНТЕРПРЕТАЦИЯ (шкала Чеддока):")
print(f"   • Связь цены с возрастом (без учета пробега):")
print(f"     {cheddock_scale(r_yx1_x2_numpy)} ({'прямая' if r_yx1_x2_numpy > 0 else 'обратная'})")
print(f"   • Связь цены с пробегом (без учета возраста):")
print(f"     {cheddock_scale(r_yx2_x1_numpy)} ({'прямая' if r_yx2_x1_numpy > 0 else 'обратная'})")

print(f"\nИнтерпретация:")
print(f"- При увеличении возраста на 1 год цена уменьшается на {abs(b_numpy[1])} тыс. руб.")
print(f"- При увеличении пробега на 1 тыс. км цена уменьшается на {abs(b_numpy[2])} тыс. руб.")
print(f"- Средняя цена нового автомобиля: {b_numpy[0]} тыс. руб.")
print(f"- Модель объясняет {R_squared_numpy*100}% вариации цены")