import numpy as np
import matplotlib.pyplot as plt

nf_count = 0
ndf_count = 0
nd2f_count = 0

def f(x):
    global nf_count
    nf_count += 1
    return 1.5 * (x + 1.5)**2 * (x - 1.1) * (x - 0.75)

def df(x):
    global ndf_count
    ndf_count += 1
    return 6 * x**3 + 5.175 * x**2 - 7.425 * x - 2.53125

def d2f(x):
    global nd2f_count
    nd2f_count += 1
    return 18 * x**2 + 10.35 * x - 7.425

def check_derivatives(x_val):
    h = 1e-5
    num_df = (f(x_val + h) - f(x_val - h)) / (2 * h)
    calc_df = df(x_val)
    print(f"--- перевірка похідних у точці x = {x_val} ---")
    print(f"аналітична f': {calc_df:.6f}, сисельна f': {num_df:.6f}, різниця: {abs(calc_df - num_df):.2e}")

#метод ньютона
def newton_method(x0, eps, max_iter=50):
    global nf_count, ndf_count, nd2f_count
    nf_count = ndf_count = nd2f_count = 0
    
    print("\nметод ньютона")
    header = f"{'k':<2} | {'xk':<9} | {'f(xk)':<9} | {'f`(xk)':<10} | {'|f`(xk)|':<10} | {'f``(xk)':<9}"
    print("-" * 65)
    print(header)
    print("-" * 65)

    x = x0
    history_x = []
    
    for k in range(max_iter):
        fx = f(x)
        d1 = df(x)
        d2 = d2f(x)
        abs_d1 = abs(d1)
        
        history_x.append((x, fx, d1, d2))
        
        print(f"{k:<2} | {x:<9.6f} | {fx:<9.6f} | {d1:<10.6f} | {abs_d1:<10.6f} | {d2:<9.6f}")
        
        if abs_d1 <= eps:
            print("-" * 65)
            print(f"break: |f'| <= eps на ітерації {k}")
            return x, history_x
            
        if d2 <= 1e-12:
            print(f"error: f'' <= 0 в x = {x:.6f}. метод йде до максимуму")
            break
            
        x = x - d1 / d2

    return x, history_x

# ньютор-рафсон
def newton_raphson_method(x0, eps, alpha=0.5, max_iter=50):
    global nf_count, ndf_count, nd2f_count
    nf_count = ndf_count = nd2f_count = 0
    
    print(f"\nметод ньютона-рафсона (alpha={alpha})")
    header = f"{'k':<2} | {'xk':<9} | {'f(xk)':<9} | {'f`(xk)':<9} | {'f``(xk)':<9} | {'t':<6} | {'x~':<9} | {'f(x~)':<9} | {'f`(x~)':<9} | умова(3.10) армійо"
    print("-" * 105)
    print(header)
    print("-" * 105)

    x = x0
    fx, d1, d2 = f(x), df(x), d2f(x)

    if abs(d1) <= eps:
        print("стартова точка вже є мінімумом")
        return x

    for k in range(max_iter):
        if d2 <= 1e-12:
             print(f"error: f'' <= 0 в x = {x:.6f}.")
             break

        t = 1.0
        while True:
            xb = x - t * d1 / d2
            d1b = df(xb)
            fb = f(xb)
            
            if abs(d1b) <= eps:
                print(f"{k:<2} | {x:<9.6f} | {fx:<9.6f} | {d1:<9.6f} | {d2:<9.6f} | {t:<6.4f} | {xb:<9.6f} | {fb:<9.6f} | {d1b:<9.6f} | stop")
                print("-" * 105)
                print(f"break: |f'(x~)| <= eps на ітерації {k}")
                return xb
            
            #умова армійо
            promised_reduction = -alpha * t * d1**2 / d2
            actual_reduction = fb - fx
            condition_met = actual_reduction <= promised_reduction
            
            cond_str = "V" if condition_met else "X"
            print(f"{k:<2} | {x:<9.6f} | {fx:<9.6f} | {d1:<9.6f} | {d2:<9.6f} | {t:<6.4f} | {xb:<9.6f} | {fb:<9.6f} | {d1b:<9.6f} | {cond_str}")
            
            if condition_met:
                x, fx, d1 = xb, fb, d1b
                d2 = d2f(x)
                break
                
            t /= 2.0
            if t < 1e-8:
                print("крок роздрібнився до мінімуму, метод застряг :(")
                return x

    return x

x_start = -2.55
eps = 0.001
alpha = 0.5

check_derivatives(x_start)

x_opt_a, hist_a = newton_method(x_start, eps)
print(f"min точка (ньютон): x* = {x_opt_a:.6f}")
print(f"звернення: Nf={nf_count}, Ndf={ndf_count}, Nd2f={nd2f_count}")

x_opt_b = newton_raphson_method(x_start, eps, alpha)
print(f"min точка (ньютон-рафсон): x* = {x_opt_b:.6f}")
print(f"звернення: Nf={nf_count}, Ndf={ndf_count}, Nd2f={nd2f_count}")


#графік
x_vals = np.linspace(-3.0, 0.0, 400)
phi_vals = [df(xv) for xv in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, phi_vals, label=r"$\phi(x) = f'(x)$", color='blue', linewidth=2)
plt.axhline(0, color='black', linewidth=1)

if len(hist_a) > 0:
    #1 ітерація
    x0, _, d1_0, d2_0 = hist_a[0]
    tangent0 = d1_0 + d2_0 * (x_vals - x0)
    plt.plot(x_vals, tangent0, '--', color='orange', label=f'дотична в $x_0={x0:.3f}$')
    plt.plot(x0, d1_0, 'bo')
    
if len(hist_a) > 1:
    #2 ітерація
    x1, _, d1_1, d2_1 = hist_a[1]
    tangent1 = d1_1 + d2_1 * (x_vals - x1)
    plt.plot(x_vals, tangent1, '--', color='green', label=f'дотична в $x1={x1:.3f}$')
    plt.plot(x1, d1_1, 'bo')

plt.plot(x_opt_a, 0, 'X', color="#e9dd3a", markersize=12, label=r"корінь $\phi(x)=0$ ($x^*$)")

plt.ylim(min(phi_vals) - 10, max(phi_vals) + 10)

plt.title(r"метод ньютона: $\phi(x) = f'(x)$ та метод дотичних")
plt.xlabel("x")
plt.ylabel(r"$\phi(x)$")
plt.grid(True)
plt.legend()
plt.show()

print("\n" + "="*50)
print("експериментище") #так він в кінці бо я зовсім за це забув, вибачте, просто закрийте графік і в терміналі виведе оцей блок
print("="*50)

bad_starts = [
    (-0.55, "права межа (b0)"), 
    (-0.99, "далеко за межами / точка перегину (f'' близька до 0)"),
    (-0.5,  "пастка (f'' < 0, напрямок до максимуму)")
]

for xs, desc in bad_starts:
    print(f"\nтест: {desc} (x0 = {xs})")
    
    print("ньютон:")
    try:
        ans_a, _ = newton_method(xs, eps, max_iter=20)
        print(f"результат: x* = {ans_a}")
    except Exception as e:
        print(f"результат: збій ({e})")
        
    print("\nньютон-рафсон:")
    try:
        ans_b = newton_raphson_method(xs, eps, alpha)
        print(f"результат: x* = {ans_b}")
    except Exception as e:
        print(f"результат: збій ({e})")