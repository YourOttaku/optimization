def f(x):
    global nf_count  #лічило
    nf_count += 1
    return 1.5 * (x + 1.5)**2 * (x - 1.1) * (x - 0.75) #ф-ція

def svenn(x0, delta0):
    global nf_count
    nf_count = 0  
    
    print("----- свенн -----")
    print(f"{'k':<3} | {'delta':<6} | {'xk':<8} | {'f(xk)':<10} | {'f(xk)<f(xk-1)':<15} | [a0, b0]")
    k, p = 0, 0
    fk = f(x0)
    print(f"{k:<3} | {delta0:<6} | {x0:<8.4f} | {fk:<10.4f} | {'-':<15} | -")

    delta = delta0
    while True:
        fk_next = f(x0 + delta)
        if fk_next < fk:
            x_prev, x_curr, f_curr = x0, x0 + delta, fk_next
            k += 1
            print(f"{k:<3} | {delta:<6} | {x_curr:<8.4f} | {f_curr:<10.4f} | {'yes':<15} | -")
            while True:
                x_next = x_curr + (2**k) * delta
                f_next = f(x_next)
                if f_next < f_curr:
                    k += 1
                    x_prev, x_curr, f_curr = x_curr, x_next, f_next
                    print(f"{k:<3} | {delta:<6} | {x_curr:<8.4f} | {f_curr:<10.4f} | {'yes':<15} | -")
                else:
                    a, b = min(x_prev, x_next), max(x_prev, x_next)
                    print(f"{k+1:<3} | {delta:<6} | {x_next:<8.4f} | {f_next:<10.4f} | {'no':<15} | [{a:.4f}, {b:.4f}]")
                    return [a, b], nf_count
        else:
            if p == 0:
                delta = -delta
                p += 1
            else:
                a, b = x0 - abs(delta), x0 + abs(delta)
                print(f"{k+1:<3} | {delta:<6} | {x0+delta:<8.4f} | {fk_next:<10.4f} | {'no':<15} | [{a:.4f}, {b:.4f}]")
                return [a, b], nf_count

def dichotomy(a, b, sigma, epsilon): #dih :sob:
    global nf_count
    nf_count = 0  
    
    print("\n----- дихотомія -----")
    print(f"{'k':<3} | {'x1':<8} | {'x2':<8} | {'f(x1)':<9} | {'f(x2)':<9} | [ak, bk]       | Lk")
    k = 0
    while True:
        Lk = b - a
        if Lk <= sigma:
            print(f"{k:<3} | {'-':<8} | {'-':<8} | {'-':<9} | {'-':<9} | [{a:.4f}, {b:.4f}] | {Lk:.4f}")
            return (a + b) / 2, nf_count
        x1 = (a + b) / 2 - epsilon / 2
        x2 = (a + b) / 2 + epsilon / 2
        f1, f2 = f(x1), f(x2)
        print(f"{k:<3} | {x1:<8.4f} | {x2:<8.4f} | {f1:<9.4f} | {f2:<9.4f} | [{a:.4f}, {b:.4f}] | {Lk:.4f}")
        if f1 < f2:
            b = x2
        else:
            a = x1
        k += 1

def half_division(a, b, sigma):
    global nf_count
    nf_count = 0  
    
    print("\n----- половинне ділення -----")
    print(f"{'k':<3} | {'x1':<8} | {'xm':<8} | {'x2':<8} | {'f(x1)':<9} | {'f(xm)':<9} | {'f(x2)':<9} | [ak, bk]       | Lk")
    k = 0
    xm = (a + b) / 2
    fm = f(xm)
    while True:
        Lk = b - a
        if Lk <= sigma:
            print(f"{k:<3} | {'-':<8} | {'-':<8} | {'-':<8} | {'-':<9} | {'-':<9} | {'-':<9} | [{a:.4f}, {b:.4f}] | {Lk:.4f}")
            return (a + b) / 2, nf_count
        x1 = a + Lk / 4
        x2 = b - Lk / 4
        f1, f2 = f(x1), f(x2)
        print(f"{k:<3} | {x1:<8.4f} | {xm:<8.4f} | {x2:<8.4f} | {f1:<9.4f} | {fm:<9.4f} | {f2:<9.4f} | [{a:.4f}, {b:.4f}] | {Lk:.4f}")
        
        if f1 < fm:
            b = xm
            xm = x1
            fm = f1
        elif f2 < fm:
            a = xm
            xm = x2
            fm = f2
        else:
            a = x1
            b = x2
        k += 1

interval, calls_svenn = svenn(-1.55, 1.0)
x_opt_dich, calls_dich = dichotomy(interval[0], interval[1], 0.001, 0.0001)
x_opt_half, calls_half = half_division(interval[0], interval[1], 0.001)

print("\n=== порівняння методів ===")
print(f"{'метод':<25} | {'точка x*':<10} | {'к-кість обчислень f(x)'}")
print("-" * 55)
print(f"{'свенн':<25} | {'-':<10} | {calls_svenn}")
print(f"{'дихотомія':<25} | {x_opt_dich:<10.5f} | {calls_dich}")
print(f"{'половинне ділення':<25} | {x_opt_half:<10.5f} | {calls_half}")