import sys

X0 = -1.55          
DELTA0 = 1.0        
EPS = 0.001        
MAX_ITER = 100


#функція
def f(x):
    return 1.5 * (x + 1.5) ** 2 * (x - 1.1) * (x - 0.75)


def df(x):
    return 1.5 * (x + 1.5) * (4 * x * x - 2.55 * x - 1.125)


#Свенн
def swann(x0, delta0):
    """Повертає (a0, b0, rows). rows - рядки табл. 4.2."""
    rows = []
    delta = delta0
    k = 0
    f0 = f(x0)
    x_prev, x_cur = x0, x0          
    fk = f0
    p = 0

    # вибір напрямку
    while True:
        x_next = x_cur + delta
        f_next = f(x_next)
        rows.append((k, delta, x_next, f_next, f_next < fk))
        if f_next < fk:
            break
        if p == 0:                   
            delta = -delta
            p += 1
            continue
        # обидва напрямки гірші: мінімум поруч із x0
        return x0 - abs(delta), x0 + abs(delta), rows

    # розширення кроку
    x_prev, x_cur, fk = x_cur, x_next, f_next
    while True:
        k += 1
        x_next = x_cur + (2 ** k) * delta
        f_next = f(x_next)
        rows.append((k, delta, x_next, f_next, f_next < fk))
        if f_next < fk:
            x_prev, x_cur, fk = x_cur, x_next, f_next
            continue
        a, b = sorted((x_prev, x_next))
        return a, b, rows


#січні
def secant(a0, b0, eps, pure=False):
    xm, xk = a0, b0
    pm, pk = df(xm), df(xk)
    n_df = 2
    if not pure and pm * pk >= 0:
        raise ValueError("f' має змінювати знак на [a0, b0]")

    rows = []
    for k in range(1, MAX_ITER + 1):
        if abs(pk - pm) < 1e-12:
            raise ZeroDivisionError("f'(xk) = f'(x{k-1}): січна горизонтальна")
        xn = (pk * xm - pm * xk) / (pk - pm)          # формула (3.16)
        pn = df(xn)
        n_df += 1
        rows.append((k, xm, pm, xk, pk, xn, pn))
        if abs(pn) <= eps:
            return xn, rows, n_df
        if pure:
            xm, pm, xk, pk = xk, pk, xn, pn
        elif pn > 0:
            xk, pk = xn, pn
        else:
            xm, pm = xn, pn
    raise RuntimeError("не збіглось за MAX_ITER ітерацій")


def print_table_swann(rows):
    print("свенн")
    print(f"{'k':>2} {'delta':>8} {'xk+1':>10} {'f(xk+1)':>12}  fk+1<f_k")
    for k, d, x, fx, ok in rows:
        print(f"{k:>2} {d:>8.3f} {x:>10.4f} {fx:>12.5f}  {'+' if ok else '-'}")


def print_table_secant(rows, title):
    print(title)
    h = ("k", "xk-1", "f'(xk-1)", "xk", "f'(xk)", "xk+1", "f'(xk+1)")
    print("{:>3} {:>10} {:>11} {:>10} {:>11} {:>10} {:>11}".format(*h))
    for r in rows:
        print("{:>3} {:>10.6f} {:>11.6f} {:>10.6f} {:>11.6f} {:>10.6f} {:>11.6f}".format(*r))


def main():
    a0, b0, sw = swann(X0, DELTA0)
    print_table_swann(sw)
    print(f"\nінтервал невизначеності [a0, b0] = [{a0:.4f}, {b0:.4f}]")
    print(f"f'(a0) = {df(a0):.5f},  f'(b0) = {df(b0):.5f}\n")

    try:
        x, rows, n_df = secant(a0, b0, EPS)
    except ValueError as e:
        print("error:", e)
        sys.exit(1)
    print_table_secant(rows, "метод січних")
    print(f"\nx* = {x:.6f}   f(x*) = {f(x):.8f}   n = {len(rows)}   "
          f"Nf' = {n_df}   (f'': 0)\n")

    xp, rows_p, n_df_p = secant(a0, b0, EPS, pure=True)
    print_table_secant(rows_p, "eксперимент - чисті січні")
    print(f"\nx* = {xp:.6f}   f(x*) = {f(xp):.8f}   n = {len(rows_p)}   "
          f"Nf' = {n_df_p}")


if __name__ == "__main__":
    main()