import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([-5.0, 7.0])
H0_LIST = [1.0, 0.7]
EPS = 0.01
H_MIN = 1e-10
MAX_ITER = 5000
METRICS = [("sphere", "сферична", "tab:blue"),
           ("cube", "кубічна", "tab:orange"),
           ("octa", "октаедрична", "tab:green")]

nf_count = 0
ng_count = 0


#ф-ція
def F(x):
    global nf_count
    nf_count += 1
    return 3 * x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 8 * x[0] - 5 * x[1]


def grad(x):
    global ng_count
    ng_count += 1
    return np.array([6 * x[0] + x[1] - 8,
                     2 * x[1] + x[0] - 5])


def check_gradient(x):
    """перевірка аналітичних похідних різницями (F(x+h e) - F(x-h e)) / 2h"""
    h = 1e-5
    num = np.zeros(2)
    for i in range(2):
        e = np.zeros(2)
        e[i] = h
        num[i] = (F(x + e) - F(x - e)) / (2 * h)
    ana = grad(x)
    print(f"перевірка похідних у {x}: аналітично {ana}, різницями {num}, "
          f"різниця {np.abs(ana - num).max():.2e}")


#напрямок
def direction(g, metric):
    if metric == "sphere": # d = -g / ||g||_2
        return -g / np.linalg.norm(g)
    if metric == "cube":   # d_i = -sign(gi)
        return -np.sign(g)
    j = np.argmax(np.abs(g)) # octa: одна координата
    d = np.zeros_like(g)
    d[j] = -np.sign(g[j])
    return d


#метод
def descent(x0, h0, eps, metric):
    """повертає (x*, рядки таблиці, траєкторія, nF, nG)"""
    global nf_count, ng_count
    nf_count = ng_count = 0

    x = x0.copy()
    Fx = F(x) # крок 1
    h = h0
    rows, path = [], [x.copy()]

    for k in range(1, MAX_ITER + 1):
        g = grad(x)# крок 2
        gn = np.linalg.norm(g)
        if gn <= eps:
            break
        d = direction(g, metric)# крок 3
        halvings = 0
        while True:
            xb = x + h * d # крок 4
            Fb = F(xb)
            if Fb < Fx: # крок 5
                break
            h /= 2
            halvings += 1
            if h < H_MIN:
                raise RuntimeError("крок здрібнів до нуля")
        rows.append((k, x[0], x[1], Fx, g[0], g[1], gn, d[0], d[1],
                     h, halvings, Fb))
        x, Fx = xb, Fb # крок 6
        path.append(x.copy())
    else:
        raise RuntimeError("перевищено MAX_ITER")
    return x, rows, np.array(path), nf_count, ng_count

def print_table(rows):
    print(f"{'k':>3} {'x1':>10} {'x2':>10} {'F(xk)':>11} {'g1':>9} {'g2':>9} "
          f"{'|g|':>8} {'d1':>7} {'d2':>7} {'h':>8} {'дробл':>5} {'F(xk+1)':>11}")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:>10.6f} {r[2]:>10.6f} {r[3]:>11.6f} "
              f"{r[4]:>9.3f} {r[5]:>9.3f} {r[6]:>8.3f} {r[7]:>7.3f} {r[8]:>7.3f} "
              f"{r[9]:>8.4f} {r[10]:>5d} {r[11]:>11.6f}")


def main():
    x_exact = np.linalg.solve(np.array([[6.0, 1.0], [1.0, 2.0]]),
                              np.array([8.0, 5.0]))
    print(f"аналітичний мінімум: x* = {x_exact}, F(x*) = {F(x_exact):.6f}\n")
    check_gradient(X0)

    summary = []
    runs = {}
    for h0 in H0_LIST:
        for metric, name, _ in METRICS:
            print(f"\n=== {name} метрика, h0 = {h0} ===")
            try:
                x, rows, path, nF, nG = descent(X0, h0, EPS, metric)
            except RuntimeError as e:
                print("помилка:", e)
                continue
            print_table(rows)
            err = np.linalg.norm(x - x_exact)
            print(f"x* = ({x[0]:.6f}; {x[1]:.6f}), F = {F(x):.6f}, "
                  f"n = {len(rows)}, NF = {nF}, NgradF = {nG}, "
                  f"||x* - xточн|| = {err:.2e}")
            summary.append((name, h0, len(rows), nF, nG, nF + nG, err))
            runs[(metric, h0)] = path

    print("\nЗведена таблиця")
    print(f"{'метрика':<12} {'h0':>4} {'n':>4} {'NF':>5} {'NgradF':>7} "
          f"{'разом':>6} {'||x*-x_точн||':>14}")
    for s in summary:
        print(f"{s[0]:<12} {s[1]:>4} {s[2]:>4} {s[3]:>5} {s[4]:>7} "
              f"{s[5]:>6} {s[6]:>14.2e}")

    # графіки
    xs = np.linspace(-7, 5, 300)
    ys = np.linspace(-2, 9, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = 3 * X ** 2 + Y ** 2 + X * Y - 8 * X - 5 * Y
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, h0 in zip(axes, H0_LIST):
        ax.contour(X, Y, Z, levels=-9 + 0.5 * 2.0 ** np.arange(0, 12), cmap="gray",
                   linewidths=0.8)
        for metric, name, color in METRICS:
            if (metric, h0) in runs:
                p = runs[(metric, h0)]
                ax.plot(p[:, 0], p[:, 1], "o-", ms=3, color=color,
                        label=f"{name} ({len(p) - 1} кроків)")
        ax.plot(*X0, "ks", label="x⁰")
        ax.plot(*x_exact, "X", ms=14, label="x*,", color="red")
        ax.set_title(f"h₀ = {h0}, ε = {EPS}")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig("trajectories.png", dpi=130)
    plt.show()


if __name__ == "__main__":
    main()