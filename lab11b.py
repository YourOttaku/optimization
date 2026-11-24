import numpy as np

from lab11a import F, grad_F, H, B, X0, cnt, fletcher_reeves

R0 = 1.0
C = 10.0
EPS = 0.01
EPS_IN = 0.01
MAX_OUTER = 30

SETS = [
    ("таблиця (обмеження з роздатки): x2 >= -1, x1 <= 1, R = 5",
     dict(c1=(-1.0, (0.0, -1.0)), c2=(-1.0, (1.0, 0.0)), R=5.0)),
    ("набір активних обмежень: x2 >= 5, x1 <= 0, R = 4",
     dict(c1=(5.0, (0.0, -1.0)), c2=(0.0, (1.0, 0.0)), R=4.0)),
]

summary = []        


# обмеження
def lin_cons(name, c0, a, kind="ineq"):
    a = np.array(a, float)
    return dict(name=name, kind=kind, c0=c0, a=a,
                g=lambda x: c0 + a @ x, gg=lambda x: a)


def circle_cons(R):
    return dict(name=f"x1^2 + x2^2 >= {R * R:g}", kind="ineq",
                g=lambda x: R * R - x @ x, gg=lambda x: -2 * x)


def viol(cons, x):
    return [max(0.0, c["g"](x)) if c["kind"] == "ineq" else c["g"](x) for c in cons]


#  Phir та Phimu
def penalty2(cons, r):
    def Phi(x):
        cnt["phi"] += 1
        return F(x) + r * sum(s * s for s in viol(cons, x))

    def gradPhi(x):
        cnt["grad"] += 1
        v = np.array(grad_F(x), float)
        for c, s in zip(cons, viol(cons, x)):
            v += 2 * r * s * c["gg"](x)        
        return v

    return Phi, gradPhi


def barrier2(cons, mu):
    # Phi_mu = F - mu*sum ln(-g_j); поза Omega (g >= 0) значення +inf
    def Phi(x):
        cnt["phi"] += 1
        gs = [c["g"](x) for c in cons]
        if max(gs) >= 0:
            return float("inf")
        return F(x) - mu * sum(np.log(-v) for v in gs)

    def gradPhi(x):
        cnt["grad"] += 1
        v = np.array(grad_F(x), float)
        for c in cons:
            v -= mu / c["g"](x) * c["gg"](x)
        return v

    return Phi, gradPhi


# внутрішні методи
def sphere_descent(Phi, gradPhi, x0, eps, h0=1.0, max_iter=50000, h_min=1e-12):
    x = np.array(x0, float)
    Fx = Phi(x)
    h = h0
    for k in range(max_iter):
        g = gradPhi(x)
        if np.linalg.norm(g) <= eps:
            return x, k, None
        d = -g / np.linalg.norm(g)
        while True:
            xb = x + h * d
            Fb = Phi(xb)
            if Fb < Fx:
                break
            h /= 2
            if h < h_min:
                raise RuntimeError("крок здрібнів до нуля")
        x, Fx = xb, Fb
    raise RuntimeError("перевищено ліміт ітерацій")


def inner_solve(method, Phi, gradPhi, x, eps_in):
    if method == "sphere":
        return sphere_descent(Phi, gradPhi, x, eps_in)
    return fletcher_reeves(Phi, gradPhi, x, eps_in)


#  метод штрафів
def sumt2(cons, x0, r0=R0, c=C, eps=EPS, eps_in=EPS_IN, method="fr", r_list=None,
          max_outer=MAX_OUTER):
    x = np.array(x0, float)
    r = r0
    outer = []
    tot_phi = tot_grad = 0
    for k in range(1, max_outer + 1):
        if r_list is not None:
            if k > len(r_list):
                break
            r = r_list[k - 1]
        cnt["phi"] = cnt["grad"] = 0
        Phi, gradPhi = penalty2(cons, r)
        start = x.copy()
        try:
            x, it, _ = inner_solve(method, Phi, gradPhi, x, eps_in)
        except RuntimeError as e:
            outer.append(dict(k=k, r=r, failed=str(e)))
            break
        tot_phi += cnt["phi"]
        tot_grad += cnt["grad"]
        s = viol(cons, x)
        P = sum(v * v for v in s)
        outer.append(dict(k=k, r=r, start=start, it=it, n_phi=cnt["phi"],
                          n_grad=cnt["grad"], x=x.copy(), F=F(x), s=s, P=P, rP=r * P,
                          lam=[2 * r * v for v in s], tot_phi=tot_phi,
                          tot_grad=tot_grad))
        if r_list is None and r * P <= eps:
            break
        r *= c
    return outer


# аналітика
def solve_kkt(lin):
    m = len(lin)
    best = None
    for mask in range(2 ** m):
        idx = [j for j in range(m) if mask >> j & 1]
        A = np.array([lin[j]["a"] for j in idx]).reshape(len(idx), 2)
        rhs = np.concatenate([B, [-lin[j]["c0"] for j in idx]])
        K = np.block([[H, A.T], [A, np.zeros((len(idx), len(idx)))]])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            continue
        x, lam = sol[:2], sol[2:]
        if all(l["c0"] + l["a"] @ x <= 1e-9 for l in lin) and np.all(lam >= -1e-9):
            lam_full = np.zeros(m)
            for j, v in zip(idx, lam):
                lam_full[j] = v
            if best is None or F(x) < F(best[0]) - 1e-12:
                best = (x, lam_full, idx)
    return best


def eq_solution(lc):
    # мінімум F на прямій c0 + a.x = 0: H x - b + lam a = 0 (lam довільного знаку)
    a = lc["a"]
    K = np.block([[H, a.reshape(2, 1)], [a.reshape(1, 2), np.zeros((1, 1))]])
    sol = np.linalg.solve(K, np.concatenate([B, [-lc["c0"]]]))
    return sol[:2], sol[2]


def eq_xr(lc, r):
    # Phi_r = F + r h^2 : (H + 2r a a^T) x = b - 2 r c0 a (без зрізки - завжди вірно)
    a = lc["a"]
    return np.linalg.solve(H + 2 * r * np.outer(a, a), B - 2 * r * lc["c0"] * a)


def barrier_xmu(lc, mu):
    # g^2 - q g - mu w = 0, g < 0;  x = H^-1 (b + mu a / g)
    a, c0 = lc["a"], lc["c0"]
    Hi = np.linalg.inv(H)
    q = c0 + a @ Hi @ B
    w = a @ Hi @ a
    g = (q - np.sqrt(q * q + 4 * mu * w)) / 2
    return Hi @ (B + mu * a / g)


def circle_minima(R):
    # локальні мінімуми F на колі радіуса R: сканування кута + уточнення
    th = np.linspace(0, 2 * np.pi, 20001)[:-1]
    f = np.array([F(R * np.array([np.cos(t), np.sin(t)])) for t in th])
    res = []
    for i in range(len(th)):
        if f[i] < f[i - 1] and f[i] <= f[(i + 1) % len(th)]:
            a, b = th[i] - 2e-3, th[i] + 2e-3
            for _ in range(80):
                m1, m2 = a + (b - a) / 3, b - (b - a) / 3
                if F(R * np.array([np.cos(m1), np.sin(m1)])) < \
                        F(R * np.array([np.cos(m2), np.sin(m2)])):
                    b = m2
                else:
                    a = m1
            t = (a + b) / 2
            x = R * np.array([np.cos(t), np.sin(t)])
            res.append((x, F(x), grad_F(x) @ x / (2 * R * R)))
    return res


def fx(x):
    return f"({x[0]:.5f}; {x[1]:.5f})"


def last_ok(outer):
    return [o for o in outer if "failed" not in o]


def add_summary(name, outer, lam_exact=None):
    ok = last_ok(outer)
    if not ok or "failed" in outer[-1]:
        summary.append((name, len(ok), sum(o["n_phi"] for o in ok),
                        sum(o["n_grad"] for o in ok), "не збіглось", "-", "-", "-"))
        return
    o = ok[-1]
    summary.append((name, len(ok), o["tot_phi"], o["tot_grad"], fx(o["x"]),
                    f"{max(max(o['s']), 0.0):.2e}", f"{o['F']:.5f}",
                    "(" + "; ".join(f"{v:.3f}" for v in o["lam"]) + ")"))


def print_outer_table(outer, ns=1, x_ref=None):
    head = f"{'r':>8} {'x(r)':>22} " + " ".join(f"{'s' + str(j + 1):>9}" for j in range(ns))
    head += f" {'F':>10} {'r*P':>9} " + " ".join(f"{'2r*s' + str(j + 1):>9}"
                                                   for j in range(ns))
    head += f" {'ітер./NPhi':>12} {'NgradPhi':>9}"
    print(head)
    for o in outer:
        if "failed" in o:
            print(f"{o['r']:>8g} {o['failed']}")
            continue
        row = f"{o['r']:>8g} {fx(o['x']):>22} " + " ".join(f"{v:>9.5f}" for v in o["s"])
        row += f" {o['F']:>10.5f} {o['rP']:>9.5f} " + " ".join(f"{v:>9.4f}"
                                                                 for v in o["lam"])
        row += f" {str(o['it']) + ' / ' + str(o['n_phi']):>12} {o['n_grad']:>9}"
        print(row)
    ok = last_ok(outer)
    if ok:
        print(f"  зовнішніх ітерацій {len(ok)}, NPhi = {ok[-1]['tot_phi']}, "
              f"NgradPhi = {ok[-1]['tot_grad']}")


# експерименти
def exp1_c(cons1, xs1, tag):
    print("\n--- 1) множник c (обмеження 1)")
    print(f"{'c':>5} {'зовн.ітер':>9} {'кінцевий r':>11} {'NPhi':>7} {'NgradPhi':>9} "
          f"{'порушення':>11} {'x':>22}")
    for c in (2, 4, 10, 100):
        out = sumt2([cons1], X0, c=c)
        o = out[-1]
        print(f"{c:>5} {len(out):>9} {o['r']:>11g} {o['tot_phi']:>7} {o['tot_grad']:>9} "
              f"{max(o['s']):>11.5f} {fx(o['x']):>22}")
        add_summary(f"c = {c}", out)
    print("висновок: малий c - багато дешевих зовнішніх ітерацій, великий c - мало "
          "ітерацій, але важча внутрішня задача на кожній")


def exp2_eps(cons1, xs1):
    print("\n--- 2) внутрішній eps ---")
    print(f"{'eps вн':>7} {'зовн.ітер':>9} {'NPhi':>7} {'NgradPhi':>9} "
          f"{'x при r = 10^4':>24} {'||x - x*||':>11} {'порушення':>11}")
    for e in (0.1, 0.01, 0.001):
        out = sumt2([cons1], X0, eps_in=e, r_list=[1, 10, 100, 1e3, 1e4])
        o = out[-1]
        print(f"{e:>7g} {len(out):>9} {o['tot_phi']:>7} {o['tot_grad']:>9} "
              f"{fx(o['x']):>24} {np.linalg.norm(o['x'] - xs1):>11.5f} "
              f"{max(o['s']):>11.5f}")
        add_summary(f"внутр. eps = {e:g}", out)
    print("порушення задає r, а не внутрішній eps")


def exp3_equality(lc1):
    print("\n--- 3) рівність h(x) = 0 замість нерівності ---")
    xe, lam_e = eq_solution(lc1)
    print(f"аналітично для рівності: x* = {fx(xe)}, F* = {F(xe):.5f}, lambda = {lam_e:.5f}")
    cons_eq = lin_cons("h", lc1["c0"], lc1["a"], kind="eq")
    rl = [1, 10, 100, 1e3, 1e4]
    out_eq = sumt2([cons_eq], X0, r_list=rl)
    out_in = sumt2([lc1], X0, r_list=rl)
    print(f"{'r':>7} {'x(r) рівність':>22} {'x(r) аналіт.':>22} {'h':>9} {'F':>10} "
          f"{'2r*h':>9} {'ітер./NPhi':>12} | {'x(r) нерівність':>22} {'g':>9}")
    for oe, oi in zip(out_eq, out_in):
        xa = eq_xr(lc1, oe["r"])
        print(f"{oe['r']:>7g} {fx(oe['x']):>22} {fx(xa):>22} {oe['s'][0]:>9.5f} "
              f"{oe['F']:>10.5f} {oe['lam'][0]:>9.4f} "
              f"{str(oe['it']) + ' / ' + str(oe['n_phi']):>12} | {fx(oi['x']):>22} "
              f"{oi['s'][0] if oi['s'][0] > 0 else lc1['g'](oi['x']):>9.5f}")
    print(f" рівність: зовнішніх ітерацій {len(out_eq)}, NPhi = {out_eq[-1]['tot_phi']}, "
          f"NgradPhi = {out_eq[-1]['tot_grad']}")
    print("зрізки немає: штраф h^2 діє з обох боків лінії, "
          "- мінімум F на самій лінії")
    add_summary("рівність h = 0, r -> 10^4", out_eq)


def exp4_barrier(cons1, ext_F):
    print("\n--- 4) лог-бар'єр зі строго внутрішнього старту ---")
    g0 = cons1["g"](X0)
    assert g0 < 0, "старт має бути строго всередині Omega"
    print(f"старт x(0) = {fx(X0)}, g(x(0)) = {g0:.4f} < 0")
    x, mu = X0.copy(), 10.0
    rows = []
    tot_phi = tot_grad = 0
    while True:
        cnt["phi"] = cnt["grad"] = 0
        Phi, gradPhi = barrier2([cons1], mu)
        x, it, _ = fletcher_reeves(Phi, gradPhi, x, EPS_IN)
        tot_phi += cnt["phi"]
        tot_grad += cnt["grad"]
        gv = cons1["g"](x)
        rows.append((mu, x.copy(), barrier_xmu(cons1, mu), gv, F(x), -mu / gv, it,
                     cnt["phi"], cnt["grad"], tot_phi))
        if mu * 1 <= EPS:
            break
        mu /= 10
    xs1 = solve_kkt([cons1])[0]
    print(f"{'mu':>8} {'x(mu) Ф-Р':>22} {'x(mu) аналіт.':>22} {'g':>10} {'F':>10} "
          f"{'F - F*':>9} {'mu/(-g)':>9} {'ітер./NPhi':>12}")
    for mu_, xx, xa, gv, Fv, lam, it, nphi, ngr, _ in rows:
        print(f"{mu_:>8g} {fx(xx):>22} {fx(xa):>22} {gv:>10.6f} {Fv:>10.5f} "
              f"{Fv - F(xs1):>9.5f} {lam:>9.4f} {str(it) + ' / ' + str(nphi):>12}")
    mu_, xx, xa, gv, Fv, lam, it, nphi, ngr, tphi = rows[-1]
    print(f"  зовнішніх ітерацій {len(rows)}, NPhi = {tphi}, NgradPhi = {tot_grad}; "
          f"усі g < 0: жодна точка не вийшла з Omega")
    print(f"двостороння оцінка F*: F(x(r)) = {ext_F:.5f} <= F* = {F(xs1):.5f} "
          f"<= F(x(mu)) = {Fv:.5f}")
    summary.append(("лог-бар'єр, mu = 10 ... 0.01", len(rows), tphi, tot_grad, fx(xx),
                    "0 (g<0)", f"{Fv:.5f}", f"({lam:.3f})"))


def exp5_two(lc1, lc2, cons1, cons2):
    print("\n--- 5) два обмеження ---")
    xs, lam, idx = solve_kkt([lc1, lc2])
    print(f"Кун-Таккер: x* = {fx(xs)}, F* = {F(xs):.5f}, lambda1 = {lam[0]:.5f}, "
          f"lambda2 = {lam[1]:.5f}, активні: {[j + 1 for j in idx]}")
    print(f"  g1(x*) = {cons1['g'](xs):.5f}, g2(x*) = {cons2['g'](xs):.5f}")
    out = sumt2([cons1, cons2], X0)
    print_outer_table(out, ns=2)
    o = out[-1]
    print(f"  оцінки 2r*sj = ({o['lam'][0]:.4f}; {o['lam'][1]:.4f}) проти "
          f"lambda = ({lam[0]:.4f}; {lam[1]:.4f})")
    if abs(cons2["g"](xs)) < 1e-9 and lam[1] < 1e-9:
        print(" обмеження 2 у розв'язку лише торкається x*: воно "
              "не діє")
    add_summary("два обмеження 1 + 2", out)


def exp6_circle(R):
    print(f"\n--- 6) крива межа: поза колом x1^2 + x2^2 >= {R * R:g}, два старти ---")
    mins = circle_minima(R)
    for x, Fv, lam in mins:
        print(f"мінімум F на колі: x = {fx(x)}, ||x|| = {np.linalg.norm(x):.5f}, "
              f"F = {Fv:.5f}, lambda = {lam:.5f}")
    cc = circle_cons(R)
    for name, st in (("старт 1", X0), ("старт 2", -X0)):
        print(f"{name}: x(0) = {fx(st)}, g = {cc['g'](st):.3f} (в Omega: g <= 0)")
        out = sumt2([cc], st)
        print(f"{'r':>8} {'x(r)':>22} {'||x(r)||':>9} {'s':>9} {'F':>10} {'r*P':>9} "
              f"{'ітер./NPhi':>12}")
        for o in out:
            print(f"{o['r']:>8g} {fx(o['x']):>22} {np.linalg.norm(o['x']):>9.5f} "
                  f"{o['s'][0]:>9.5f} {o['F']:>10.5f} {o['rP']:>9.5f} "
                  f"{str(o['it']) + ' / ' + str(o['n_phi']):>12}")
        print(f"  зовнішніх ітерацій {len(out)}, NPhi = {out[-1]['tot_phi']}, "
              f"NgradPhi = {out[-1]['tot_grad']}")
        add_summary(f"коло, {name}", out)
    fs = [(x, Fv) for x, Fv, _ in mins]
    if len(fs) > 1:
        best = min(fs, key=lambda t: t[1])
        print(f"vінімумів на колі {len(fs)}: глобальний {fx(best[0])}, "
              f"F = {best[1]:.5f}; порівняти F з різних стартів")
    else:
        print("на колі один мінімум F (F не парна), обидва старти мають прийти до нього")


def exp7_inner(cons1):
    print("\n--- 7) внутрішній метод ---")
    print("старт x(0) для кожного r; h0 = 1 з дробленням, критерій ||grad Phi|| <= 0.01")
    print(f"{'метод':<22} {'r':>6} {'ітер':>7} {'NPhi':>8} {'NgradPhi':>10} "
          f"{'порушення':>11}")
    for method, name in (("sphere", "сферична"), ("fr", "Флетчер-Рівс")):
        for r in (1, 10, 100):
            out = sumt2([cons1], X0, method=method, r_list=[r])
            o = out[-1]
            if "failed" in o:
                print(f"{name:<22} {r:>6} не збіглось: {o['failed']}")
                continue
            print(f"{name:<22} {r:>6} {o['it']:>7} {o['n_phi']:>8} {o['n_grad']:>10} "
                  f"{max(o['s']):>11.5f}")
            if r == 100:
                add_summary(f"{name}, r = 100", out)


# main
def run_set(title, p):
    print("\n" + "=" * 110)
    print(title)
    lc1 = lin_cons("g1", p["c1"][0], p["c1"][1])
    lc2 = lin_cons("g2", p["c2"][0], p["c2"][1])
    xs1, lam1, idx1 = solve_kkt([lc1])
    print(f"обмеження 1: g1 = {p['c1'][0]:g} + ({p['c1'][1][0]:g}, {p['c1'][1][1]:g}).x <= 0; "
          f"x* = {fx(xs1)}, F* = {F(xs1):.5f}, lambda = {lam1[0]:.5f} "
          f"({'ACTIVE' if idx1 else 'INACTIVE'})")
    summary.clear()
    exp1_c(lc1, xs1, title)
    exp2_eps(lc1, xs1)
    exp3_equality(lc1)
    ext = sumt2([lc1], X0)[-1]
    exp4_barrier(lc1, ext["F"])
    exp5_two(lc1, lc2, lc1, lc2)
    exp6_circle(p["R"])
    exp7_inner(lc1)

    print("\n--- 8) зведена таблиця прогонів ---")
    print(f"{'прогін':<34} {'зовн.':>5} {'NPhi':>7} {'NgradPhi':>9} {'відповідь':>22} "
          f"{'порушення':>10} {'F':>10} {'lambda':>16}")
    for s in summary:
        print(f"{s[0]:<34} {s[1]:>5} {s[2]:>7} {s[3]:>9} {s[4]:>22} {s[5]:>10} "
              f"{s[6]:>10} {s[7]:>16}")


def main():
    for title, p in SETS:
        run_set(title, p)


if __name__ == "__main__":
    main()