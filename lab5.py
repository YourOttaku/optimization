import numpy as np
import matplotlib.pyplot as plt

nf_count = 0

def f(x):
    global nf_count
    nf_count += 1
    return 1.5 * (x + 1.5)**2 * (x - 1.1) * (x - 0.75)

def powell_method(x_start, dx, sigma_x, sigma_f, max_iter=50):
    global nf_count
    nf_count = 0
    
    print("--- метод пауелла ---")
    
    x1 = x_start
    x2 = x1 + dx
    f1, f2 = f(x1), f(x2)
    
    if f1 > f2:
        x3 = x1 + 2 * dx
    else:
        x3 = x1 - dx
    f3 = f(x3)
    
    print(f"стартові точки: x1={x1:.4f}, x2={x2:.4f}, x3={x3:.4f}")
    
    header = f"{'k':<2} | {'x1':<8} | {'x2':<8} | {'x3':<8} | {'f(x1)':<9} | {'f(x2)':<9} | {'f(x3)':<9} | {'a1':<9} | {'a2':<9} | {'x~':<8} | {'f(x~)':<10} | {'|fmin-f~|':<11} | {'|xmin-x~|':<11}"
    print("-" * 135)
    print(header)
    print("-" * 135)

    x_list = [x1, x2, x3]
    f_list = [f1, f2, f3]
    
    history_x = list(x_list)
    history_f = list(f_list)
    parabola_points = []
    first_iter_parabola = None

    for k in range(1, max_iter + 1):
        i_min = np.argmin(f_list)
        i_max = np.argmax(f_list)
        x_min, f_min_val = x_list[i_min], f_list[i_min]
        x_max, f_max_val = x_list[i_max], f_list[i_max]
        
        curr_x1, curr_x2, curr_x3 = x_list
        curr_f1, curr_f2, curr_f3 = f_list
        
        # Захист від ділення на нуль
        if abs(curr_x2 - curr_x1) < 1e-12 or abs(curr_x3 - curr_x2) < 1e-12 or abs(curr_x3 - curr_x1) < 1e-12:
            print("error: точки збіглися, неможливо побудувати параболу")
            break
            
        a1 = (curr_f2 - curr_f1) / (curr_x2 - curr_x1)
        a2 = ((curr_f3 - curr_f1) / (curr_x3 - curr_x1) - a1) / (curr_x3 - curr_x2)
        
        if a2 > 0:
            x_tilde = -a1 / (2 * a2) + (curr_x1 + curr_x2) / 2
        else:
            # Захист від горба (a2 <= 0)
            x_tilde = 2 * x_min - x_max
            
        f_tilde = f(x_tilde)
        history_x.append(x_tilde)
        history_f.append(f_tilde)
        
        if k == 1:
            first_iter_parabola = (curr_f1, a1, a2, curr_x1, curr_x2)
            parabola_points = [curr_x1, curr_x2, curr_x3]

        delta_f = abs(f_min_val - f_tilde)
        delta_x = abs(x_min - x_tilde)
        
        row_str = f"{k:<2} | {curr_x1:<8.5f} | {curr_x2:<8.5f} | {curr_x3:<8.5f} | {curr_f1:<9.5f} | {curr_f2:<9.5f} | {curr_f3:<9.5f} | {a1:<9.5f} | {a2:<9.5f} | {x_tilde:<8.5f} | {f_tilde:<10.5f} | {delta_f:<11.6f} | {delta_x:<11.6f}"
        print(row_str)

        if delta_f <= sigma_f and delta_x <= sigma_x:
            print("-" * 135)
            print(f"зупинка на ітерації {k}: обидві умови виконані")
            break
            
        x_list[i_max] = x_tilde
        f_list[i_max] = f_tilde

    x_opt = x_tilde
    print(f"\nточка мінімуму (пауелл): x* = {x_opt:.6f}")
    print(f"к-кість обчислень функції: Nf = {nf_count}")
    
    return x_opt, nf_count, history_x, history_f, first_iter_parabola, parabola_points

x_start = -2.55 
dx = 0.5      
sigma_x = 0.001
sigma_f = 0.01

x_opt, n_f, h_x, h_f, par_params, p_points = powell_method(x_start, dx, sigma_x, sigma_f)

#графік
x_vals = np.linspace(-3.0, 0.0, 400)
y_vals = [1.5 * (x + 1.5)**2 * (x - 1.1) * (x - 0.75) for x in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='$f(x)$', color='blue', linewidth=2)

if par_params:
    f1, a1, a2, x1, x2 = par_params
    y_par = f1 + a1 * (x_vals - x1) + a2 * (x_vals - x1) * (x_vals - x2)
    mask = (y_par < max(y_vals)) & (y_par > min(y_vals) - 5)
    plt.plot(x_vals[mask], y_par[mask], '--', color='orange', label='парабола 1-ї ітерації $y(x)$')
    
    for px in p_points:
        plt.plot(px, 1.5 * (px + 1.5)**2 * (px - 1.1) * (px - 0.75), 'bo', markersize=8)
        plt.text(px, 1.5 * (px + 1.5)**2 * (px - 1.1) * (px - 0.75) + 0.5, f"{px:.2f}", ha='center')

plt.plot(x_opt, 1.5 * (x_opt + 1.5)**2 * (x_opt - 1.1) * (x_opt - 0.75), 'X', color="#e9dd3a", markersize=12, label='знайдений мінімум $x^*$')

plt.title('метод Пауелла: функція та параболічна апроксимація')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()