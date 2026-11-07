import numpy as np
import matplotlib.pyplot as plt

EPS = 0.01
N_VARS = 2                 
SVENN_DELTA = 0.01
GOLDEN_EPS = 1e-6
MAX_ITER = 5000

nf_count = 0
ng_count = 0

def F(x):
    global nf_count
    nf_count += 1
    return 3 * x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 8 * x[0] - 5 * x[1]


def grad(x):
    global ng_count
    ng_count += 1
    return np.array([6 * x[0] + x[1] - 8,
                     2 * x[1] + x[0] - 5])


H = np.array([[6.0, 1.0], [1.0, 2.0]]) # гесіян (сталий)
X0 = np.array([-5.0, 7.0])


#функція Розенброка
def F_ros(x):
    global nf_count
    nf_count += 1
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_ros(x):
    global ng_count
    ng_count += 1
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                     200 * (x[1] - x[0] ** 2)])


#одновимірний пошук
def svenn(phi, h0=0.0, delta=SVENN_DELTA):
    """локалізація мінімуму: повертає [a, b]"""
    f0 = phi(h0)
    if phi(h0 + delta) >= f0:
        if phi(h0 - delta) >= f0:
            return h0 - delta, h0 + delta
        delta = -delta
    x_prev, x_cur, f_cur = h0, h0 + delta, phi(h0 + delta)
    k = 1
    while True:
        x_next = x_cur + (2 ** k) * delta
        f_next = phi(x_next)
        if f_next >= f_cur:
            return tuple(sorted((x_prev, x_next)))
        x_prev, x_cur, f_cur = x_cur, x_next, f_next
        k += 1


def golden(phi, a, b, eps=GOLDEN_EPS):
    """золотий переріз: повертає h*"""
    r = (np.sqrt(5) - 1) / 2            
    x1 = b - r * (b - a)
    x2 = a + r * (b - a)
    f1, f2 = phi(x1), phi(x2)
    while b - a > eps:
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - r * (b - a)
            f1 = phi(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + r * (b - a)
            f2 = phi(x2)
    return (a + b) / 2


def line_search(Ffun, x, d):
    phi = lambda h: Ffun(x + h * d)
    a, b = svenn(phi)
    return golden(phi, a, b)


def fletcher_reeves(Ffun, gfun, x0, eps=EPS, n=N_VARS, restart=True,
                    use_beta=True, max_iter=MAX_ITER):
    global nf_count, ng_count
    nf_count = ng_count = 0

    x = np.array(x0, float)
    g = gfun(x) # крок 2
    d = -g
    k = 0
    rows, path, checks = [], [x.copy()], {}

    while np.linalg.norm(g) > eps: # крок 3
        if k >= max_iter:
            break
        h = line_search(Ffun, x, d) # крок 4
        x_new = x + h * d                        # крок 5
        g_new = gfun(x_new)# крок 6
        beta = (g_new @ g_new) / (g @ g) if use_beta else 0.0
        if restart and (k + 1) % n == 0:
            beta = 0.0
        # значення F для таблиці — не рахуються у лічильник звернень
        saved = nf_count
        Fx, Fn = Ffun(x), Ffun(x_new)
        nf_count = saved
        rows.append((k, x.copy(), Fx, g.copy(), np.linalg.norm(g), d.copy(),
                     h, beta, Fn))
        if k == 0:
            checks["perp"] = g_new @ d  # <g(1), d(0)>
            checks["conj"] = (-g_new + (g_new @ g_new) / (g @ g) * d) @ H @ d \
                if Ffun is F else None  # d(1)^T H d(0)
        d_new = -g_new + beta * d   # крок 7
        if k == 0:
            checks["descent"] = g_new @ d_new # <g(1), d(1)>
            checks["d1"] = d_new.copy()
            checks["beta1"] = (g_new @ g_new) / (g @ g)
        x, g, d = x_new, g_new, d_new
        path.append(x.copy())
        k += 1
    return x, rows, np.array(path), nf_count, ng_count, checks


def print_table(rows):
    print(f"{'k':>3} {'x(k)':>22} {'F(x(k))':>11} {'gradF':>22} {'|gradF|':>9} "
          f"{'d(k)':>22} {'hk':>9} {'betak+1':>9} {'F(x(k+1))':>12}")
    for k, x, Fx, g, gn, d, h, beta, Fn in rows:
        print(f"{k:>3} ({x[0]:>9.6f};{x[1]:>9.6f}) {Fx:>11.6f} "
              f"({g[0]:>9.6f};{g[1]:>9.6f}) {gn:>9.6f} "
              f"({d[0]:>9.6f};{d[1]:>9.6f}) {h:>9.6f} {beta:>9.6f} {Fn:>12.4e}")


def main():
    x_exact = np.linalg.solve(H, np.array([8.0, 5.0]))
    print(f"гесіян H = {H.tolist()},  аналітичний мінімум x* = {x_exact}, "
          f"F(x*) = {F(x_exact):.6f}")
    print(f"∇F(x(0)) = {grad(X0)}, |∇F| = {np.linalg.norm(grad(X0)):.6f}\n")

    #перевірка лінійного пошуку на першому кроці (параболa)
    g0 = grad(X0)
    d0 = -g0
    h_exact = -(g0 @ d0) / (d0 @ H @ d0)
    phi = lambda h: F(X0 + h * d0)
    a, b = svenn(phi)
    h_num = golden(phi, a, b)
    print(f"лінійний пошук, крок 1: Свенн -> [{a:.4f}; {b:.4f}], "
          f"золотий h = {h_num:.6f}, точно h* = -<∇F,d>/dᵀHd = {h_exact:.6f}\n")

    print("=== Флетчер–Рівс, точний пошук, перезапуск кожні n = 2 ===")
    x, rows, path_fr, nF, nG, ch = fletcher_reeves(F, grad, X0)
    print_table(rows)
    err = np.linalg.norm(x - x_exact)
    print(f"x* = ({x[0]:.6f}; {x[1]:.6f}), n = {len(rows)}, NF = {nF}, "
          f"NgradF = {nG}, ||x* - xточн|| = {err:.2e}")
    print("\nперевірки (після 1-ї ітерації):")
    print(f"  1. <∇F(x(1)), d(0)>     = {ch['perp']:.3e}   (перпендикулярність)")
    print(f"  2. d(1)ᵀ·H·d(0)         = {ch['conj']:.3e}   (спряженість)")
    print(f"  3. <∇F(x(1)), d(1)>     = {ch['descent']:.6f}   (< 0 — напрямок спуску)")
    print(f"     beta_1 = {ch['beta1']:.6f},  d(1) = {ch['d1']}")
    n_fr, nF_fr, nG_fr, err_fr = len(rows), nF, nG, err

    print("\n=== антиградієнт та точний пошук (beta = 0) ===")
    xa, rows_a, path_ag, nFa, nGa, _ = fletcher_reeves(F, grad, X0, use_beta=False)
    for r in rows_a:
        print(f"{r[0]:>3} ({r[1][0]:>10.6f};{r[1][1]:>10.6f})  F = {r[2]:>12.6f}")
    print(f"x* = ({xa[0]:.6f}; {xa[1]:.6f}), n = {len(rows_a)}, NF = {nFa}, "
          f"NgradF = {nGa}")

    xr0 = np.array([-1.2, 1.0])
    print("\n=== Розенброк, x(0) = (-1.2; 1), eps = 0.01 ===")
    ros = {}
    for label, rs in (("з перезапуском кожні 2", True), ("без перезапуску", False)):
        xx, rr, pp, a1, a2, _ = fletcher_reeves(F_ros, grad_ros, xr0, restart=rs)
        print(f"Флетчер–Рівс {label}: x = ({xx[0]:.5f}; {xx[1]:.5f}), "
              f"n = {len(rr)}, NF = {a1}, NgradF = {a2}, F = {F_ros(xx):.2e}")
        ros[label] = (len(rr), a1, a2)

    # сферична метрика 8лр
    path_sph, nF_s, nG_s = sphere_8(X0)

    # таблиця
    print("\nзведена таблиця")
    print(f"{'метод':<30} {'n':>4} {'NF':>5} {'NgradF':>7} {'разом':>6}")
    print(f"{'Флетчер–Рівс':<30} {n_fr:>4} {nF_fr:>5} {nG_fr:>7} {nF_fr + nG_fr:>6}")
    print(f"{'антиградієнт + точний пошук':<30} {len(rows_a):>4} {nFa:>5} "
          f"{nGa:>7} {nFa + nGa:>6}")
    print(f"{'сферична, дроблення':<30} {len(path_sph) - 1:>4} {nF_s:>5} "
          f"{nG_s:>7} {nF_s + nG_s:>6}")

    # -------- графік
    xs = np.linspace(-8, 6, 400)
    ys = np.linspace(-3, 9, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = 3 * X ** 2 + Y ** 2 + X * Y - 8 * X - 5 * Y
    plt.figure(figsize=(9, 7))
    plt.contour(X, Y, Z, levels=-9 + 0.5 * 2.0 ** np.arange(0, 12),
                colors="gray", linewidths=0.8)
    plt.plot(path_sph[:, 0], path_sph[:, 1], "o-", ms=3, color="tab:blue",
             label=f"сферична: {len(path_sph) - 1} кроків")
    plt.plot(path_ag[:, 0], path_ag[:, 1], "s--", ms=4, color="tab:purple",
             label=f"антиградієнт + точний пошук: {len(path_ag) - 1} кроків")
    plt.plot(path_fr[:, 0], path_fr[:, 1], "o-", ms=6, lw=2.2, color="tab:green",
             label=f"Флетчер–Рівс: {len(path_fr) - 1} кроки")
    plt.plot(*X0, "ks", label="x(0)")
    plt.plot(*x_exact, "X", ms=14, label="x*", color="#FFE600")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("лінії рівня F та траєкторії")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=130)
    plt.show()


def sphere_8(x0, h0=1.0, eps=EPS):
    """сферична метрика з дробленням кроку — для порівняння"""
    global nf_count, ng_count
    nf_count = ng_count = 0
    x = np.array(x0, float)
    Fx = F(x)
    h = h0
    path = [x.copy()]
    while True:
        g = grad(x)
        gn = np.linalg.norm(g)
        if gn <= eps:
            break
        d = -g / gn
        while True:
            xb = x + h * d
            Fb = F(xb)
            if Fb < Fx:
                break
            h /= 2
        x, Fx = xb, Fb
        path.append(x.copy())
    return np.array(path), nf_count, ng_count


if __name__ == "__main__":
    main()