import math
import matplotlib.pyplot as plt
import numpy as np

nf_count = 0 

def f(x):
    global nf_count
    nf_count += 1
    return 1.5 * (x + 1.5)**2 * (x - 1.1) * (x - 0.75)

# переріз
def golden_section_search(a, b, epsilon):
    global nf_count
    nf_count = 0
    a0, b0 = a, b
    
    probed_points = [] 
    
    print("--- метод золотого перерізу ---")
    print(f"{'k':<3} | {'ak':<8} | {'bk':<8} | {'x1':<8} | {'x2':<8} | {'f(x1)':<9} | {'f(x2)':<9} | Lk      | Рішення")
    
    tau = (math.sqrt(5) - 1) / 2
    
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    probed_points.append((x1, f1))
    probed_points.append((x2, f2))
    
    k = 1
    while True:
        Lk = b - a
        if Lk <= epsilon:
            print("-" * 90)
            print(f"зупинка: Lk ({Lk:.5f}) <= epsilon ({epsilon})")
            
            eta = (Lk / (b0 - a0)) ** (1 / nf_count)
            return (a + b) / 2, nf_count, eta, probed_points
            
        print(f"{k:<3} | {a:<8.5f} | {b:<8.5f} | {x1:<8.5f} | {x2:<8.5f} | {f1:<9.5f} | {f2:<9.5f} | {Lk:.5f} | ", end="")
        
        if f1 <= f2:
            print("f1 <= f2 -> b=x2")
            b = x2
            x2 = x1
            f2 = f1  
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1) 
            probed_points.append((x1, f1)) 
        else:
            print("f1 > f2 -> a=x1")
            a = x1
            x1 = x2
            f1 = f2  
            x2 = a + tau * (b - a)
            f2 = f(x2) 
            probed_points.append((x2, f2))
        k += 1

# дихотомія для порівняння
def dichotomy_silent(a, b, epsilon, delta_x=0.0001):
    global nf_count
    nf_count = 0
    a0, b0 = a, b
    while (b - a) > epsilon:
        x1 = (a + b) / 2 - delta_x / 2
        x2 = (a + b) / 2 + delta_x / 2
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
    Ln = b - a
    eta = (Ln / (b0 - a0)) ** (1 / nf_count)
    return nf_count, eta

# половинне ділення для порівняння
def half_division_silent(a, b, epsilon):
    global nf_count
    nf_count = 0
    a0, b0 = a, b
    xm = (a + b) / 2
    fm = f(xm)
    while (b - a) > epsilon:
        L = b - a
        x1 = a + L / 4
        x2 = b - L / 4
        f1, f2 = f(x1), f(x2)
        if f1 < fm:
            b, xm, fm = xm, x1, f1
        elif f2 < fm:
            a, xm, fm = xm, x2, f2
        else:
            a, b = x1, x2
    Ln = b - a
    eta = (Ln / (b0 - a0)) ** (1 / nf_count)
    return nf_count, eta

a0 = -2.55
b0 = -0.55
epsilon_val = 0.05  

x_opt_gold, nf_gold, eta_gold, probed_pts = golden_section_search(a0, b0, epsilon_val)
nf_dich, eta_dich = dichotomy_silent(a0, b0, epsilon_val)
nf_half, eta_half = half_division_silent(a0, b0, epsilon_val)

print(f"точка мінінуму золотого перерізу: x* = {x_opt_gold:.5f}")

# Фінальна таблиця порівняння
print("\n-порівняння методів-")
print(f"L0 = {b0 - a0}, epsilon = {epsilon_val}")
print("-" * 45)
print(f"{'метод':<25} | {'Nf':<5} | {'η':<10}")
print("-" * 45)
print(f"{'метод дихотомії':<25} | {nf_dich:<5} | {eta_dich:.4f}")
print(f"{'половинне ділення':<25} | {nf_half:<5} | {eta_half:.4f}")
print(f"{'золотий переріз':<25} | {nf_gold:<5} | {eta_gold:.4f}")
print("-" * 45)

# графік
x_vals = np.linspace(a0 - 0.2, b0 + 0.2, 400)
y_vals = [1.5 * (x + 1.5)**2 * (x - 1.1) * (x - 0.75) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x) = 1.5(x+1.5)^2(x-1.1)(x-0.75)", color="#1f77b4", linewidth=2)

px = [p[0] for p in probed_pts]
py = [p[1] for p in probed_pts]

plt.scatter(px, py, color="red", zorder=5, s=40, label="Пробні точки")

y_opt = 1.5 * (x_opt_gold + 1.5)**2 * (x_opt_gold - 1.1) * (x_opt_gold - 0.75)
plt.scatter([x_opt_gold], [y_opt], color="#e9dd3a", marker="X", s=250, zorder=6, label="мінімум x*")
plt.axvline(a0, color='gray', linestyle='--', alpha=0.7, label="межі a0, b0")
plt.axvline(b0, color='gray', linestyle='--', alpha=0.7)
plt.title("метод золотого перерізу", fontsize=14, pad=15)
plt.xlabel("x", fontsize=12)
plt.ylabel("f(x)", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()