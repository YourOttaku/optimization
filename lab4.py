import math
nf_count = 0

def f(x):
    global nf_count
    nf_count += 1
    return 1.5 * (x + 1.5)**2 * (x - 1.1) * (x - 0.75)

def fibonacci_search(a, b, epsilon):
    global nf_count
    nf_count = 0
    a0, b0 = a, b
    L0 = b - a
    R = L0 / epsilon

    F = [0, 1, 1]
    while F[-1] < R:
        F.append(F[-1] + F[-2])
    N = len(F) - 1

    delta = epsilon / 10

    print("--- метод чисел фібоначчі ---")
    print(f"L0 = {L0:.4f}, R = {R:.4f} -> N = {N}, F[N] = {F[N]}")
    print(f"{'k':<2} | {'m':<2} | {'Fm-2/Fm':<7} | {'Fm-1/Fm':<7} | {'ak':<8} | {'bk':<8} | {'x1':<8} | {'x2':<8} | {'f(x1)':<9} | {'f(x2)':<9} | Lk      | Рішення")

    x1 = a + (F[N-2] / F[N]) * L0
    x2 = a + (F[N-1] / F[N]) * L0
    f1, f2 = f(x1), f(x2)

    for k in range(N - 2):
        m = N - k
        Lk = b - a
        frac1 = f"{F[m-2]}/{F[m]}"
        frac2 = f"{F[m-1]}/{F[m]}"
        
        row_str = f"{k:<2} | {m:<2} | {frac1:<7} | {frac2:<7} | {a:<8.5f} | {b:<8.5f} | {x1:<8.5f} | {x2:<8.5f} | {f1:<9.5f} | {f2:<9.5f} | {Lk:.5f} | "

        if f1 <= f2:
            print(row_str + "f1 <= f2 -> b=x2")
            b = x2
            x2 = x1
            f2 = f1  
            if m == 3:
                pass 
            elif m - 1 == 3: 
                x1 = x2 - delta 
                f1 = f(x1)
            else:
                x1 = a + (F[m-3] / F[m-1]) * (b - a)
                f1 = f(x1)
        else:
            print(row_str + "f1 > f2 -> a=x1")
            a = x1
            x1 = x2
            f1 = f2  
            if m == 3:
                pass #
            elif m - 1 == 3: 
                x2 = x1 + delta 
                f2 = f(x2)
            else:
                x2 = a + (F[m-2] / F[m-1]) * (b - a)
                f2 = f(x2)

    Lk_final = b - a
    print("-" * 115)
    
    #гарантія
    guarantee = L0 / F[N]
    if Lk_final <= guarantee + delta:
        print(f"зупинка: Lкін = {Lk_final:.5f} <= L0/FN + зсув ({guarantee:.5f} + {delta}) ---> гарантія є")
    else:
        print(f"зупинка: Lкін = {Lk_final:.5f} > L0/FN + зсув ({guarantee:.5f} + {delta}) ---> гарантія порушена")

    #похибка
    x_opt = (a + b) / 2
    x_exact = -1.5 
    error = abs(x_opt - x_exact)
    print(f"похибка |x* - x*точн| = {error:.5f}")

    eta = (Lk_final / (b0 - a0)) ** (1 / nf_count)
    return x_opt, nf_count, eta

def dichotomy_silent(a, b, epsilon, delta_x=0.0001):
    global nf_count
    nf_count = 0
    a0, b0 = a, b
    while (b - a) > epsilon:
        x1 = (a + b) / 2 - delta_x / 2
        x2 = (a + b) / 2 + delta_x / 2
        if f(x1) < f(x2): b = x2
        else: a = x1
    eta = ((b - a) / (b0 - a0)) ** (1 / nf_count)
    return nf_count, eta

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
        if f1 < fm: b, xm, fm = xm, x1, f1
        elif f2 < fm: a, xm, fm = xm, x2, f2
        else: a, b = x1, x2
    eta = ((b - a) / (b0 - a0)) ** (1 / nf_count)
    return nf_count, eta

def golden_section_silent(a, b, epsilon):
    global nf_count
    nf_count = 0
    a0, b0 = a, b
    tau = (math.sqrt(5) - 1) / 2
    x1 = a + (1 - tau) * (b - a)
    x2 = a + tau * (b - a)
    f1, f2 = f(x1), f(x2)
    while (b - a) > epsilon:
        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - tau) * (b - a)
            f1 = f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + tau * (b - a)
            f2 = f(x2)
    eta = ((b - a) / (b0 - a0)) ** (1 / nf_count)
    return nf_count, eta

a0 = -2.55
b0 = -0.55
epsilon_val = 0.05  

x_opt_fib, nf_fib, eta_fib = fibonacci_search(a0, b0, epsilon_val)
nf_dich, eta_dich = dichotomy_silent(a0, b0, epsilon_val)
nf_half, eta_half = half_division_silent(a0, b0, epsilon_val)
nf_gold, eta_gold = golden_section_silent(a0, b0, epsilon_val)

print(f"\nточка мінімуму методу фібоначчі: x* = {x_opt_fib:.5f}")
print("\n-порівняння методів-")
print(f"L0 = {b0 - a0}, epsilon = {epsilon_val}")
print("-" * 45)
print(f"{'метод':<25} | {'Nf':<5} | {'η':<10}")
print("-" * 45)
print(f"{'метод дихотомії':<25} | {nf_dich:<5} | {eta_dich:.4f}")
print(f"{'половинне ділення':<25} | {nf_half:<5} | {eta_half:.4f}")
print(f"{'золотий переріз':<25} | {nf_gold:<5} | {eta_gold:.4f}")
print(f"{'метод фібоначчі':<25} | {nf_fib:<5} | {eta_fib:.4f}")
print("-" * 45)