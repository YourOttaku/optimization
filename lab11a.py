import numpy as np
import matplotlib.pyplot as plt

R0 = 1.0
C = 10.0
EPS = 0.01
EPS_IN = 0.01
SVENN_DELTA = 0.01
GOLDEN_EPS = 1e-6
RESTART_N = 2
MAX_INNER = 3000
MAX_OUTER = 30

H = np.array([[6.0, 1.0], [1.0, 2.0]])
B = np.array([8.0, 5.0])
X0 = np.array([-5.0, 7.0])

CONS_TABLE = {"name": "x2 >= -1   (g = -1 - x2 <= 0)  - з таблиці варіантів",
              "c0": -1.0, "a": np.array([0.0, -1.0])}

cnt = {"phi": 0, "grad": 0}


# F та обмеження
def F(x):
    return 3 * x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 8 * x[0] - 5 * x[1]


def grad_F(x):
    return np.array([6 * x[0] + x[1] - 8, 2 * x[1] + x[0] - 5])


def make_cons(cd):
    # повертає список [(g, gradg)] для штрафної функції
    a, c0 = cd["a"], cd["c0"]
    return [(lambda x: c0 + a @ x, lambda x: a)]


def g_of(cd, x):
    return cd["c0"] + cd["a"] @ x


# штрафна функція
def penalty(Ff, gradFf, cons, r):
    def Phi(x):
        cnt["phi"] += 1
        return Ff(x) + r * sum(max(0.0, g(x)) ** 2 for g, _ in cons)

    def gradPhi(x):
        cnt["grad"] += 1
        v = np.array(gradFf(x), float)
        for g, gg in cons:
            s = max(0.0, g(x))
            v += 2 * r * s * np.array(gg(x), float)
        return v

    return Phi, gradPhi


#лінійний пошук
def svenn(phi, h0=0.0, delta=SVENN_DELTA):
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


def line_search(Phi, x, d):
    phi = lambda h: Phi(x + h * d)
    a, b = svenn(phi)
    return golden(phi, a, b)


# Флетчер-Рівс 
def fletcher_reeves(Phi, gradPhi, x0, eps=EPS_IN, n=RESTART_N):
    x = np.array(x0, float)
    g = gradPhi(x)
    d = -g
    k = 0
    rows = []
    while np.linalg.norm(g) > eps:
        if k >= MAX_INNER:
            raise RuntimeError("внутрішня задача не збіглась")
        h = line_search(Phi, x, d)
        x_new = x + h * d
        g_new = gradPhi(x_new)
        beta = (g_new @ g_new) / (g @ g)
        if (k + 1) % n == 0:
            beta = 0.0
        # значення Phi для таблиці не потрапляють у лічильник звернень
        saved = cnt["phi"]
        rows.append((k, x.copy(), Phi(x), g.copy(), np.linalg.norm(g), d.copy(),
                     h, beta, Phi(x_new)))
        cnt["phi"] = saved
        x, g, d = x_new, g_new, -g_new + beta * d
        k += 1
    return x, k, rows


# метод штрафів (SUMT)
def sumt(Ff, gradFf, cons, x0, r0=R0, c=C, eps=EPS, max_outer=MAX_OUTER):
    x = np.array(x0, float)
    r = r0
    outer, traj = [], [x.copy()]
    inner_first = None
    total_phi = total_grad = 0
    for k in range(1, max_outer + 1):
        cnt["phi"] = cnt["grad"] = 0
        Phi, gradPhi = penalty(Ff, gradFf, cons, r)
        start = x.copy()
        x, it, rows = fletcher_reeves(Phi, gradPhi, x)
        if inner_first is None:
            inner_first = rows
        total_phi += cnt["phi"]
        total_grad += cnt["grad"]
        gs = [g(x) for g, _ in cons]
        P = sum(max(0.0, gv) ** 2 for gv in gs)
        outer.append(dict(k=k, r=r, start=start, it=it, n_phi=cnt["phi"],
                          n_grad=cnt["grad"], x=x.copy(), F=Ff(x), g=gs[0],
                          P=P, rP=r * P, lam=2 * r * max(0.0, gs[0]),
                          total=total_phi, total_grad=total_grad))
        traj.append(x.copy())
        if r * P <= eps:
            break
        r *= c
    return outer, traj, inner_first


# аналітика для контролю
def exact_solution(cd):
    xu = np.linalg.solve(H, B)
    if g_of(cd, xu) <= 0:
        return xu, 0.0, False
    a = cd["a"]
    K = np.block([[H, a.reshape(2, 1)], [a.reshape(1, 2), np.zeros((1, 1))]])
    sol = np.linalg.solve(K, np.concatenate([B, [-cd["c0"]]]))
    return sol[:2], sol[2], True


def analytic_xr(cd, r):
    a, c0 = cd["a"], cd["c0"]
    x = np.linalg.solve(H + 2 * r * np.outer(a, a), B - 2 * r * c0 * a)
    return x if g_of(cd, x) > 0 else None


def polish(x, cd):
    gv = g_of(cd, x)
    if gv <= 0:
        return x.copy()
    a = cd["a"]
    return x - gv * a / (a @ a)


def print_inner(rows):
    print(f"{'k':>2} {'x(k)':>24} {'Phi':>11} {'gradPhi':>24} {'|gradPhi|':>10} "
          f"{'d(k)':>24} {'hk':>9} {'beta':>9} {'Phi(next)':>11}")
    for k, x, Px, g, gn, d, h, beta, Pn in rows:
        print(f"{k:>2} ({x[0]:>9.6f};{x[1]:>9.6f}) {Px:>11.6f} "
              f"({g[0]:>9.4f};{g[1]:>9.4f}) {gn:>10.6f} "
              f"({d[0]:>9.4f};{d[1]:>9.4f}) {h:>9.6f} {beta:>9.6f} {Pn:>11.6f}")


def print_outer(outer, cd):
    print(f"{'k':>2} {'r':>8} {'старт внутрішньої':>20} {'ітер':>5} {'NPhi':>6} "
          f"{'NgradPhi':>9} {'x(r)':>20} {'F(x(r))':>10} {'g(x(r))':>10} "
          f"{'r*P':>9} {'2r*g':>8} {'NPhi разом':>11} {'x(r) аналіт.':>22}")
    for o in outer:
        xa = analytic_xr(cd, o["r"])
        sxa = "g<0: не працює" if xa is None else f"({xa[0]:.6f};{xa[1]:.6f})"
        print(f"{o['k']:>2} {o['r']:>8g} ({o['start'][0]:>8.4f};{o['start'][1]:>8.4f}) "
              f"{o['it']:>5} {o['n_phi']:>6} {o['n_grad']:>9} "
              f"({o['x'][0]:>8.5f};{o['x'][1]:>8.5f}) {o['F']:>10.5f} {o['g']:>10.6f} "
              f"{o['rP']:>9.5f} {o['lam']:>8.4f} {o['total']:>11} {sxa:>22}")


def run_case(cd, tag):
    print("=" * 100)
    print(f"{tag}: обмеження {cd['name']}")
    xs, lam, active = exact_solution(cd)
    xu = np.linalg.solve(H, B)
    print(f"безумовний мінімум xu = {xu}, g(xu) = {g_of(cd, xu):.4f}, "
          f"g(x(0)) = {g_of(cd, X0):.4f}")
    print(f"аналітично: x* = ({xs[0]:.6f}; {xs[1]:.6f}), F* = {F(xs):.6f}, "
          f"lambda = {lam:.6f}, обмеження {'ACTIVE' if active else 'INACTIVE'}")
    if not active:
        print("ALERT: безумовний мінімум допустимий, тому штраф ніколи не вмикається "
              "(P = 0, lambda = 0).")
    cons = make_cons(cd)
    outer, traj, inner = sumt(F, grad_F, cons, X0)

    print(f"\nвнутрішня задача для r = {outer[0]['r']:g} :")
    print_inner(inner)
    print("\nпротокол зовнішніх ітерацій:")
    print_outer(outer, cd)
    last = outer[-1]
    print(f"\nстоп при r = {last['r']:g}: x* ~ ({last['x'][0]:.4f}; {last['x'][1]:.4f}), "
          f"F = {last['F']:.4f} (точно {F(xs):.4f}), порушення max(0,g) = "
          f"{max(0.0, last['g']):.2e}, lambda ~ {last['lam']:.4f} (точно {lam:.4f}), "
          f"NPhi = {last['total']}, NgradPhi = {last['total_grad']}")
    if active:
        print("перевірка: порушення * 2r = lambda ->  "
              + ", ".join(f"r={o['r']:g}: {max(0, o['g']) * 2 * o['r']:.4f}" for o in outer))

    # дошліфовування проєкцією 
    xh = polish(last["x"], cd)
    print(f"\nдошліфовування: x^ = ({xh[0]:.6f}; {xh[1]:.6f}), g(x^) = "
          f"{g_of(cd, xh):.2e}, F(x^) = {F(xh):.6f}, "
          f"||x^ - x*|| = {np.linalg.norm(xh - xs):.2e}")

    # порівняння: одразу великий r зі старту 
    print("\nпорівняння з одразу великим r (зі старту x(0)):")
    print(f"{'варіант':<34} {'ітер':>6} {'NPhi':>8} {'NgradPhi':>10}")
    print(f"{'r=1 -> 10 -> ... теплий старт':<34} {sum(o['it'] for o in outer):>6} "
          f"{last['total']:>8} {last['total_grad']:>10}")
    for rr in (1e4, 1e6):
        cnt["phi"] = cnt["grad"] = 0
        Phi, gradPhi = penalty(F, grad_F, cons, rr)
        try:
            _, it, _ = fletcher_reeves(Phi, gradPhi, X0)
            print(f"{'одразу r = ' + format(rr, 'g'):<34} {it:>6} {cnt['phi']:>8} "
                  f"{cnt['grad']:>10}")
        except RuntimeError:
            print(f"{'одразу r = ' + format(rr, 'g'):<34} не збіглось за {MAX_INNER} ітер")

    plot_case(cd, outer, traj, xs, xu, active, "trajectories.png")
    return outer


# графіки
def plot_case(cd, outer, traj, xs, xu, active, fname):
    traj = np.array(traj)
    X, Y = np.meshgrid(np.linspace(-8, 6, 400), np.linspace(-3, 9, 400))
    Z = 3 * X ** 2 + Y ** 2 + X * Y - 8 * X - 5 * Y
    G = cd["c0"] + cd["a"][0] * X + cd["a"][1] * Y

    fig0, (a0, a1) = plt.subplots(1, 2, figsize=(14, 6))
    a0.contour(X, Y, Z, levels=-9 + 0.5 * 2.0 ** np.arange(0, 12), colors="gray",
               linewidths=0.8, linestyles="solid")
    a0.contourf(X, Y, G, levels=[-1e9, 0], colors=["tab:green"], alpha=0.15)
    a0.contour(X, Y, G, levels=[0], colors="darkgreen", linewidths=2)
    a0.plot(traj[:, 0], traj[:, 1], "o-", color="tab:blue", ms=5, label="x(r), r = 1 ... rk")
    a0.plot(*X0, "ks", label="x(0)")
    a0.plot(*xu, "ro", ms=10, label="безумовний мінімум")
    a0.plot(*xs, "x", color="gold", mec="k", ms=18, label="умовний мінімум x*")
    a0.set_title("лінії рівня F, допустима множина Ω (зелена) і траєкторія x(r)")
    a0.set_xlabel("x₁")
    a0.set_ylabel("x₂")
    a0.grid(True, alpha=0.3)
    a0.legend()
    rs = np.array([o["r"] for o in outer])
    gs = np.array([o["g"] for o in outer])
    if active:
        a1.loglog(rs, np.maximum(gs, 1e-16), "o-", label="g(x(r))")
        a1.loglog(rs, cd_lambda_line(cd, rs, outer), "k--", label="λ/(2r)")
        a1.set_ylabel("порушення g(x(r))")
    else:
        a1.semilogx(rs, gs, "o-", label="g(x(r))")
        a1.axhline(0, color="darkgreen", lw=2)
        a1.text(0.5, 0.5, "g < 0: обмеження неактивне, штраф = 0",
                transform=a1.transAxes, ha="center")
        a1.set_ylabel("g(x(r))")
    a1.set_title("порушення обмеження залежно від r")
    a1.set_xlabel("r (логарифмічна шкала)")
    a1.grid(True, alpha=0.3, which="both")
    a1.legend()
    
    fig0.tight_layout()
    fig0.savefig("contours_trajectory.png", dpi=130)
    plt.show()


def cd_lambda_line(cd, rs, outer):
    """теоретична залежність порушення від r: lambda/(2r)"""
    lam = exact_solution(cd)[1]
    return lam / (2 * rs)


def main():
    run_case(CONS_TABLE, "var 17, частина 1")


if __name__ == "__main__":
    main()