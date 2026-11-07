import numpy as np
import matplotlib.pyplot as plt

EPS = 0.01
X0 = np.array([-5.0, 7.0])
X_EXACT = np.array([1.0, 2.0])

nf_count = 0


def F(x):
    global nf_count
    nf_count += 1
    return 3 * x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 8 * x[0] - 5 * x[1]


def G(x):
    """негладка функція зі слайда 12: x* = (1; -2), G(x*) = 0"""
    global nf_count
    nf_count += 1
    return 2 * abs(x[0] - 1) + 3 * abs(x[1] + 2) + 0.5 * (x[0] - 1) ** 2


def ROS(x):
    """функція Розенброка: x* = (1; 1)"""
    global nf_count
    nf_count += 1
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# дослідний пошук
def explore(Ffun, y, Fy, d):
    z, Fz = y.copy(), Fy
    probes = []
    for i in range(len(z)):
        t = z.copy()
        t[i] += d
        Ft = Ffun(t)
        probes.append((t.copy(), Ft))
        if Ft < Fz:
            z, Fz = t, Ft
            continue
        t = z.copy()
        t[i] -= d
        Ft = Ffun(t)
        probes.append((t.copy(), Ft))
        if Ft < Fz:
            z, Fz = t, Ft
    return z, Fz, probes


# метод Хуке-Джівса
def hooke_jeeves(Ffun, x0, d0=1.0, eps=EPS, max_iter=100000, verbose=True):
    global nf_count
    nf_count = 0

    base = np.array(x0, float) # кроки 1-2
    Fb = Ffun(base)
    y, Fy, d, from_pattern = base.copy(), Fb, d0, False
    k = 0
    bases = [base.copy()]
    log = []

    for _ in range(max_iter):
        n_before = nf_count
        z, Fz, probes = explore(Ffun, y, Fy, d)# крок 3
        n_probes = nf_count - n_before
        kind = "дослідний зі зразка" if from_pattern else "дослідний із бази"
        log.append((k, kind, y.copy(), Fy, d, z.copy(), Fz, n_probes, nf_count))

        if Fz < Fb: # крок 4
            new_base, F_new = z, Fz # крок 5
            y = z + (z - base)
            n_before = nf_count
            Fy = Ffun(y)
            base, Fb = new_base, F_new
            k += 1
            bases.append(base.copy())
            from_pattern = True
            log.append((k, "нова база -> зразок", base.copy(), Fb, d, y.copy(),
                        Fy, nf_count - n_before, nf_count))
        elif from_pattern:# крок 6
            y, Fy, from_pattern = base.copy(), Fb, False
            log.append((k, "повернення в базу", base.copy(), Fb, d, None,
                        None, 0, nf_count))
        else: # крок 7
            d /= 2
            log.append((k, "дроблення Δ", base.copy(), Fb, d, None, None, 0,
                        nf_count))
            if d < eps:
                return base, Fb, nf_count, np.array(bases), log
    raise RuntimeError("перевищено max_iter")


def fmt(v):
    return "—" if v is None else f"({v[0]:.4f}; {v[1]:.4f})"


def print_protocol(log):
    print(f"{'k':>2} {'дія':<22} {'звідки':>20} {'F':>10} {'Δ':>8} "
          f"{'куди':>20} {'F':>10} {'проб':>5} {'NF':>5}")
    for k, act, frm, Ff, d, to, Ft, pr, nf in log:
        print(f"{k:>2} {act:<22} {fmt(frm):>20} {Ff:>10.5f} {d:>8.4f} "
              f"{fmt(to):>20} {('—' if Ft is None else format(Ft, '.5f')):>10} "
              f"{pr:>5} {nf:>5}")


def contour_data():
    xs = np.linspace(-8, 6, 400)
    ys = np.linspace(-3, 9, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = 3 * X ** 2 + Y ** 2 + X * Y - 8 * X - 5 * Y
    return X, Y, Z


def main():
    print(f"F(x(0)) = {F(X0):.4f}; аналітичний мінімум x* = {X_EXACT}, "
          f"F(x*) = {F(X_EXACT):.4f} (лише для контролю)\n")

    results = {}
    for d0 in (1.0, 0.7):
        print(f"=== прогін Δ0 = {d0}, ε = {EPS} ===")
        x, Fx, nf, bases, log = hooke_jeeves(F, X0, d0)
        print_protocol(log)
        err = np.linalg.norm(x - X_EXACT)
        print(f"x* = ({x[0]:.6f}; {x[1]:.6f}), F = {Fx:.3e}, "
              f"баз = {len(bases) - 1}, NF = {nf}, NgradF = 0, "
              f"||x* - xточн|| = {err:.2e}\n")
        results[d0] = (x, Fx, nf, bases, err)

    print("=== негладка функція G, старт (5; -3), Δ0 = 1 ===")
    xg, Fg, nfg, bg, lg = hooke_jeeves(G, np.array([5.0, -3.0]), 1.0)
    print_protocol(lg)
    print(f"x* = ({xg[0]:.6f}; {xg[1]:.6f}), G = {Fg:.3e}, NG = {nfg}\n")

    print("=== Розенброк, старт (-1.2; 1) ===")
    ros = []
    for eps_r in (0.01, 1e-4):
        xr, Fr, nfr, br, _ = hooke_jeeves(ROS, np.array([-1.2, 1.0]), 1.0,
                                          eps=eps_r)
        print(f"Δ0 = 1, ε = {eps_r:g}: x = ({xr[0]:.4f}; {xr[1]:.4f}), "
              f"f = {Fr:.3e}, баз = {len(br) - 1}, NF = {nfr}")
        ros.append((eps_r, len(br) - 1, nfr, xr, Fr))

    #зведена таблиця
    print("\nзведена таблиця (наша F, x(0) = (-5; 7), ε = 0.01)")
    print(f"{'метод':<34} {'n':>4} {'звернень':>9} {'потрібно':>12}")
    x1, _, nf1, b1, e1 = results[1.0]
    x7, _, nf7, b7, e7 = results[0.7]
    rows = [
        ("сферична, дроблення h", 14, 21 + 15, "∇F"),
        ("кубічна, дроблення h", 6, 7 + 7, "sign ∇F"),
        ("октаедрична, дроблення h", 11, 12 + 12, "max|∂F/∂xi|"),
        ("Флетчер–Рівс", 2, 75 + 3, "∇F, d(k-1)"),
        ("Хук–Джівс, Δ0 = 1", len(b1) - 1, nf1, "лише F"),
        ("Хук–Джівс, Δ0 = 0.7", len(b7) - 1, nf7, "лише F"),
    ]
    for r in rows:
        print(f"{r[0]:<34} {r[1]:>4} {r[2]:>9} {r[3]:>12}")
    print("(для градієнтних методів: звернень = NF + NgradF; "
          "для Хука–Джівса — лише NF)")

    # графік 
    X, Y, Z = contour_data()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, d0, color in zip(axes, (1.0, 0.7), ("tab:red", "tab:cyan")):
        x, _, nf, bases, _ = results[d0]
        ax.contour(X, Y, Z, levels=-9 + 0.5 * 2.0 ** np.arange(0, 12),
                   colors="gray", linewidths=0.8, linestyles="solid")
        ax.plot(bases[:, 0], bases[:, 1], "o-", color=color, ms=5,
                label=f"базові точки Хука–Джівса ({len(bases) - 1}), NF = {nf}")
        ax.plot(*X0, "ks", label="x(0)")
        ax.plot(*X_EXACT, "X", ms=14, label="x*", color="#FFE600")
        ax.set_title(f"Δ₀ = {d0}, ε = {EPS}")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()