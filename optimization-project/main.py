import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from src.variant import (
    STUDENT_ID, SEED, NOISE_SIGMA, X0, DELTA0, EPS, SIMPLEX_STEP, LAMBDA0, SIZE
)
from src.rng import Rng
from src.image import clean_image, add_noise, save_png
from src.metrics import mse, psnr
from src.oracle import Oracle
from src.filters import gaussian_blur, bilateral_filter
from src.lab1_svenn import svenn
from src.lab2_search import dichotomy, golden_section
from src.lab3_nelder_mead import nelder_mead
from src.lab3_powell import powell, line_minimize
from src.operators import (
    grad_xy, divergence, energy, energy_grad, tikhonov_exact
)
from src.lab4_gradient import gradient_descent
from src.lab5_constrained import (
    project, violation, projected_gradient, penalty_method
)


def main():
    os.makedirs("outputs", exist_ok=True)

    # =========================================================================
    print("=" * 75)
    print(f"підготовка(0) (варіант {STUDENT_ID}, SEED = {SEED})")
    print("=" * 75)

    rng_test = Rng(SEED)
    first6 = [round(rng_test.normal(), 6) for _ in range(6)]
    print(f"перші 6 нормальних чисел Rng({SEED}): {first6}")

    clean = clean_image(SIZE)
    raw, noisy = add_noise(clean, NOISE_SIGMA, SEED)
    save_png(clean, "outputs/00_clean.png")
    save_png(noisy, "outputs/00_noisy.png")

    print("\nчисте зображення:")
    for i in range(60, 65):
        print("  " + "  ".join(f"{clean[i][j]:.4f}" for j in range(60, 65)))

    print("\nзіпсоване зображення noisy:")
    for i in range(60, 65):
        print("  " + "  ".join(f"{noisy[i][j]:.4f}" for j in range(60, 65)))

    v_raw = violation(raw)
    mse_noisy = mse(noisy, clean)
    psnr_noisy = psnr(noisy, clean)
    print(f"\nпікселів поза [0, 1] у raw: {v_raw['count']} із {SIZE*SIZE} "
          f"({100.0 * v_raw['count'] / (SIZE*SIZE):.2f}%)")
    print(f"найменше у raw: {raw.min():.4f}, найбільше у raw: {raw.max():.4f}")
    print(f"стартові метрики: MSE = {mse_noisy:.8f}, PSNR = {psnr_noisy:.4f} дБ")

    # =========================================================================
    
    print("\n" + "=" * 75)
    print("алгоритм свенна(1)")
    print("=" * 75)

    r_x0 = int(4.0 * X0 + 0.5)
    t_x0 = np.arange(-r_x0, r_x0 + 1, dtype=float)
    k_x0 = np.exp(-(t_x0 * t_x0) / (2.0 * X0 * X0))
    k_x0 /= k_x0.sum()
    print(f"ядро гауса для X0 = {X0} (r={r_x0}): {[round(float(x), 6) for x in k_x0]}")
    blurred_x0 = gaussian_blur(noisy, X0)
    print(f"піксель (62, 62) після розмиття з X0={X0}: {blurred_x0[62, 62]:.6f}")

    LO1, HI1 = 0.05, 8.0
    def f1(sigma):
        clamped = min(max(float(sigma), LO1), HI1)
        return mse(gaussian_blur(noisy, clamped), clean) + 1000.0 * (float(sigma) - clamped) ** 2

    oracle1 = Oracle(f1)
    res_svenn = svenn(oracle1, X0, DELTA0, LO1, HI1)

    print(f"\n{'k':>2} {'x_k':>9} {'f(xk)':>12} {'Deltak':>9} {'p':>2}  коментар")
    for r in res_svenn["rows"]:
        print(f"{r['k']:>2} {r['x']:>9.4f} {r['f']:>12.8f} {r['delta']:>9.4f} {r['p']:>2}  {r['comment']}")
    a0, b0 = res_svenn["a"], res_svenn["b"]
    L0 = b0 - a0
    print(f"\nзнайдений відрізок: [{a0:.4f}, {b0:.4f}], довжина = {L0:.4f}, Nf = {res_svenn['n_f']}")

    # Графік Лаб. 1
    grid1 = np.linspace(LO1, HI1, 200)
    vals1 = [oracle1.peek(s) for s in grid1]
    plt.figure(figsize=(9, 5))
    plt.plot(grid1, vals1, label="f(σ)")
    plt.scatter([r["x"] for r in res_svenn["rows"]],
                [r["f"] for r in res_svenn["rows"]],
                color="red", zorder=5, label="пробні точки свенна")
    plt.axvspan(a0, b0, alpha=0.2, color="green",
                label=f"знайдений відрізок [{a0:.3f}, {b0:.3f}]")
    plt.xscale("log")
    plt.xlabel("σ — сила розмиття")
    plt.ylabel("f(σ) = MSE")
    plt.title("локалізація мінімуму методом свенна")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("outputs/lab1_svenn.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n__обов'язкові експерименти (1)__")
    for label, x_init, d_init in [
        ("дослід 1 (delta0 / 10)", X0, DELTA0 / 10.0),
        ("дослід 2 (10 * delta0)", X0, DELTA0 * 10.0),
        ("дослід 3 (x0 = 3.0)", 3.0, DELTA0),
    ]:
        oracle1.reset()
        r_exp = svenn(oracle1, x_init, d_init, LO1, HI1)
        print(f"{label:22s}: [{r_exp['a']:.4f}, {r_exp['b']:.4f}], "
              f"L = {r_exp['b'] - r_exp['a']:.4f}, p = {r_exp['p']}, Nf = {r_exp['n_f']}")

    # =========================================================================

    print("\n" + "=" * 75)
    print("метод дихотомії та золотий переріз(2)")
    print("=" * 75)

    oracle1.reset()
    res_d = dichotomy(oracle1, a0, b0, EPS)
    oracle1.reset()
    res_g = golden_section(oracle1, a0, b0, EPS)

    for name, res in [("дихотомія", res_d), ("золотий переріз", res_g)]:
        L_end = res["b"] - res["a"]
        comp = (L_end / L0) ** (1.0 / res["n_f"])
        print(f"{name:18s}: σ* = {res['x']:.6f}, f(σ*) = {res['f']:.8f}, "
              f"Lкінц = {L_end:.3e}, ітер = {res['iterations']:2d}, "
              f"Nf = {res['n_f']:2d}, стиснення/звернення = {comp:.4f}")

    sigma_star = res_g["x"]
    restored2 = gaussian_blur(noisy, sigma_star)
    save_png(restored2, "outputs/lab2_restored.png")
    psnr_lab2 = psnr(restored2, clean)
    print(f"відновлення: MSE = {mse(restored2, clean):.8f}, PSNR = {psnr_lab2:.4f} дБ")

    print("\n__обов'язкові експерименти (2)__")
    oracle1.reset()
    rd_e6 = dichotomy(oracle1, a0, b0, 1e-6)
    oracle1.reset()
    rg_e6 = golden_section(oracle1, a0, b0, 1e-6)
    print(f"дослід 1 (eps=1e-6): дихотомія -> ітер={rd_e6['iterations']}, Nf={rd_e6['n_f']}; "
          f"золотий переріз -> ітер={rg_e6['iterations']}, Nf={rg_e6['n_f']}")
    print("дослід 3 (ручний підбір σ):")
    for s_man in (0.3, 0.5, 0.8, 1.5):
        img_man = gaussian_blur(noisy, s_man)
        print(f"  σ = {s_man:.1f} -> MSE = {mse(img_man, clean):.8f}, PSNR = {psnr(img_man, clean):.4f} дБ")

    # =========================================================================

    print("\n" + "=" * 75)
    print("методи нелдера-міда та пауелла(3)")
    print("=" * 75)

    LO2 = np.array([0.3, 0.01])
    HI2 = np.array([5.0, 1.0])
    def f2(theta):
        point = np.asarray(theta, dtype=float)
        clamped = np.clip(point, LO2, HI2)
        value = mse(bilateral_filter(noisy, clamped[0], clamped[1]), clean)
        gap = point - clamped
        return value + 1000.0 * float(gap @ gap)

    sigma_s0 = min(max(2.0 * sigma_star, 0.5), 4.0)
    sigma_r0 = min(max(0.15 + 0.1 * NOISE_SIGMA, 0.02), 0.9)
    theta0 = np.array([sigma_s0, sigma_r0])

    oracle2 = Oracle(f2)
    print(f"початкова точка θ0 = ({sigma_s0:.6f}, {sigma_r0:.6f}), f(θ0) = {oracle2.peek(theta0):.8f}")

    oracle2.reset()
    res_nm = nelder_mead(oracle2, theta0, SIMPLEX_STEP)
    oracle2.reset()
    res_pw = powell(oracle2, theta0)

    print(f"нелдер—мід : θ* = ({res_nm['x'][0]:.6f}, {res_nm['x'][1]:.6f}), "
          f"f* = {res_nm['f']:.10f}, ітер = {res_nm['iterations']}, Nf = {res_nm['n_f']}")
    print(f"пауелл     : θ* = ({res_pw['x'][0]:.6f}, {res_pw['x'][1]:.6f}), "
          f"f* = {res_pw['f']:.10f}, циклів = {res_pw['cycles']}, Nf = {res_pw['n_f']}")

    theta_best = res_nm["x"] if res_nm["f"] <= res_pw["f"] else res_pw["x"]
    restored3 = bilateral_filter(noisy, theta_best[0], theta_best[1])
    save_png(restored3, "outputs/lab3_restored.png")
    psnr_lab3 = psnr(restored3, clean)
    print(f"відновлення: MSE = {mse(restored3, clean):.8f}, PSNR = {psnr_lab3:.4f} дБ")

    #графік до 3 частини
    ss = np.linspace(0.3, 5.0, 40)
    sr = np.linspace(0.01, 1.0, 40)
    grid2 = np.array([[oracle2.peek([a, b]) for a in ss] for b in sr])
    plt.figure(figsize=(9, 6))
    levels = np.quantile(grid2, np.linspace(0.0, 0.6, 25))
    plt.contourf(ss, sr, grid2, levels=levels, alpha=0.75)
    plt.colorbar(label="f(σs, σr) = MSE")
    nm_path = np.array([theta0] + [r["x"] for r in res_nm["rows"]])
    pw_path = np.array([theta0] + [r["x"] for r in res_pw["rows"]])
    plt.plot(nm_path[:, 0], nm_path[:, 1], "o-", ms=3, lw=1, label="нелдер—мід")
    plt.plot(pw_path[:, 0], pw_path[:, 1], "s-", ms=5, lw=1.5, label="пауелл")
    plt.scatter([theta_best[0]], [theta_best[1]], marker="*", s=220, zorder=5, label="оптимум")
    plt.xlabel("σs — просторовий масштаб")
    plt.ylabel("σr — фотометричний поріг")
    plt.title("частина 3: лінії рівня і траєкторії методів")
    plt.legend()
    plt.savefig("outputs/lab3_contours.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n__Обов'язкові експерименти (3)__")
    for s_mult, s_val in [("s/3", SIMPLEX_STEP / 3.0), ("s", SIMPLEX_STEP), ("3s", SIMPLEX_STEP * 3.0)]:
        oracle2.reset()
        r_s = nelder_mead(oracle2, theta0, s_val)
        print(f"дослід 1 ({s_mult:3s} = {s_val:.4f}): θ* = ({r_s['x'][0]:.4f}, {r_s['x'][1]:.4f}), "
              f"f* = {r_s['f']:.8f}, ітер = {r_s['iterations']}, Nf = {r_s['n_f']}")

    oracle2.reset()
    r_pw_strict = powell(oracle2, theta0, tol=1e-7)
    print(f"дослід 3 (пауелл tol=1e-7): циклів = {r_pw_strict['cycles']}, "
          f"Nf = {r_pw_strict['n_f']}, f* = {r_pw_strict['f']:.10f}")

    fixed_sr = theta_best[1]
    oracle_1d_ss = Oracle(lambda s_s: f2([s_s, fixed_sr]))
    r_1d_ss = golden_section(oracle_1d_ss, 0.3, 5.0, eps=1e-5)
    print(f"дослід 4 (фіксоване σr*={fixed_sr:.6f}): σ_s* = {r_1d_ss['x']:.6f}, "
          f"f* = {r_1d_ss['f']:.10f}, Nf = {r_1d_ss['n_f']}")

    # =========================================================================

    print("\n" + "=" * 75)
    print("градієнтні методи(4)")
    print("=" * 75)

    #чек операторів
    rng_np = np.random.default_rng(0)
    u_t = rng_np.normal(size=(9, 7))
    px_t = rng_np.normal(size=(9, 7))
    py_t = rng_np.normal(size=(9, 7))
    gx_t, gy_t = grad_xy(u_t)
    adj_err = abs(float(np.sum(gx_t * px_t + gy_t * py_t)) - (-float(np.sum(u_t * divergence(px_t, py_t)))))
    print(f"перевірка 1 (нев'язка спряженості): {adj_err:.3e}")

    I_t = rng_np.random((16, 16))
    V_t = rng_np.normal(size=(16, 16))
    noisy_small = noisy[:16, :16]
    t_eps = 1e-6
    num_d = (energy(I_t + t_eps * V_t, noisy_small, 0.5) - energy(I_t - t_eps * V_t, noisy_small, 0.5)) / (2 * t_eps)
    ana_d = float(np.sum(energy_grad(I_t, noisy_small, 0.5) * V_t))
    print(f"перевірка 2 (відносна різниця градієнта): {abs(num_d - ana_d) / abs(ana_d):.3e}")

    # lambda*
    def g_lam(lam):
        c_lam = min(max(float(lam), 1e-4), 50.0)
        I_lam = tikhonov_exact(noisy, c_lam)
        return mse(I_lam, clean) + 1000.0 * (float(lam) - c_lam) ** 2

    oracle_lam = Oracle(g_lam)
    br_lam = svenn(oracle_lam, LAMBDA0, 0.2, 1e-4, 50.0)
    n_f_svenn_lam = oracle_lam.n_f
    res_lam = golden_section(oracle_lam, br_lam["a"], br_lam["b"], eps=1e-3)
    lam_star = res_lam["x"]

    L = 1.0 + 8.0 * lam_star
    mu = 1.0
    kappa = L / mu
    h_opt = 1.0 / L
    beta_hb = ((math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))) ** 2
    beta_nag_sc = (math.sqrt(L) - math.sqrt(mu)) / (math.sqrt(L) + math.sqrt(mu))

    print(f"\ncвенн по λ: [{br_lam['a']:.6f}, {br_lam['b']:.6f}] ({n_f_svenn_lam} звернень)")
    print(f"pолотий переріз: λ* = {lam_star:.6f}, g(λ*) = {res_lam['f']:.8f} "
          f"({res_lam['iterations']} ітерацій, {oracle_lam.n_f} звернень разом)")
    print(f"константи: L = {L:.6f}, κ = {kappa:.6f}, h = 1/L = {h_opt:.6f}, "
          f"βHB = {beta_hb:.6f}, βNAGsc = {beta_nag_sc:.6f}")

    I_star = tikhonov_exact(noisy, lam_star)
    E_star = energy(I_star, noisy, lam_star)
    grad_star = float(np.linalg.norm(energy_grad(I_star, noisy, lam_star)))
    print(f"еталон I*: E* = {E_star:.8f}, ‖∇E(I*)‖ = {grad_star:.3e}")

    # запуски
    configs = [
        ("спуск, h=1/L",              dict(kind="gd",     beta=None)),
        ("спуск, армійо",             dict(kind="armijo", beta=None)),
        (f"важка кулька, β={beta_hb:.4f}", dict(kind="hb",     beta=beta_hb)),
        ("важка кулька, β=0.9",       dict(kind="hb",     beta=0.9)),
        ("нестеров, опуклий",         dict(kind="nag",    beta=None)),
        ("нестеров, сильно опуклий",  dict(kind="nag",    beta=beta_nag_sc)),
    ]

    results4 = {}
    print(f"\n{'метод':28s} | {'ітер':>4s} | {'Nf':>4s} | {'N∇f':>4s} | {'разом':>5s} | {'E(IN)':>12s} | {'‖∇E‖':>9s} | Збігся")
    print("-" * 88)
    for name, kw in configs:
        res = gradient_descent(noisy, lam_star, L, x_ref=I_star, **kw)
        results4[name] = res
        tot = res["n_f"] + res["n_grad"]
        print(f"{name:28s} | {res['iterations']:4d} | {res['n_f']:4d} | {res['n_grad']:4d} | "
              f"{tot:5d} | {res['E']:12.8f} | {res['grad']:9.2e} | {'так' if res['converged'] else 'ні'}")

    #графік до частини 4
    plt.figure(figsize=(9, 5))
    for name, res in results4.items():
        ys = [max(r["E"] - E_star, 1e-16) for r in res["rows"]]
        plt.semilogy(range(len(ys)), ys, label=name)
    plt.xlabel("номер ітерації k")
    plt.ylabel("E(Ik) − E*")
    plt.title("частина 4: збіжність за значенням функції")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("outputs/lab4_conv_f.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    for name, res in results4.items():
        ys = [max(r["dist"], 1e-16) for r in res["rows"]]
        plt.semilogy(range(len(ys)), ys, label=name)
    plt.xlabel("номер ітерації k")
    plt.ylabel("‖Ik − I*‖")
    plt.title("частина 4: збіжність за відстанню до рішення")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("outputs/lab4_conv_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    restored4 = np.clip(results4["нестеров, сильно опуклий"]["I"], 0.0, 1.0)
    save_png(restored4, "outputs/lab4_restored.png")
    psnr_lab4 = psnr(restored4, clean)
    print(f"\nPSNR попіксельного відновлення (4) = {psnr_lab4:.4f} дБ")

    print("\n__обов'язкові експерименти (4)__")
    lam_high = 6.0
    L_high = 1.0 + 8.0 * lam_high
    b_hb_high = ((math.sqrt(L_high) - 1.0) / (math.sqrt(L_high) + 1.0)) ** 2
    b_nag_high = (math.sqrt(L_high) - 1.0) / (math.sqrt(L_high) + 1.0)
    r_gd_49 = gradient_descent(noisy, lam_high, L_high, kind="gd", max_iter=3000)
    r_hb_49 = gradient_descent(noisy, lam_high, L_high, kind="hb", beta=b_hb_high, max_iter=3000)
    r_nag_49 = gradient_descent(noisy, lam_high, L_high, kind="nag", beta=b_nag_high, max_iter=3000)
    print(f"дослід 1 (λ=6, κ=49): GD -> {r_gd_49['iterations']} ітер, "
          f"HB -> {r_hb_49['iterations']} ітер, NAG_sc -> {r_nag_49['iterations']} ітер")

    r_div = gradient_descent(noisy, lam_star, L, kind="gd", step=2.1 / L, max_iter=50)
    e_div = {r["k"]: round(r["E"], 2) for r in r_div["rows"] if r["k"] in (0, 10, 20, 30, 40)}
    print(f"дослід 2 (h = 2.1/L, розбіжність): E по ітераціях = {e_div}")

    print("дослід 3 (вплив λ на PSNR):")
    for l_val in (0.05, round(lam_star, 4), 1.0, 5.0):
        img_l = np.clip(tikhonov_exact(noisy, l_val), 0.0, 1.0)
        print(f"  λ = {l_val:<6.4f} -> PSNR = {psnr(img_l, clean):.4f} дБ")

    # =========================================================================

    print("\n" + "=" * 75)
    print("умовна оптимізація")
    print("=" * 75)

    unconstrained = tikhonov_exact(raw, lam_star)
    v_unc = violation(unconstrained)
    print(f"безумовний розв'язок на raw: поза Ω = {v_unc['count']} пікселів "
          f"({100.0 * v_unc['count'] / (SIZE*SIZE):.2f}%), "
          f"макс. порушення = {v_unc['max']:.6f}, E = {energy(unconstrained, raw, lam_star):.8f}")

    #гістограма до частини 5
    plt.figure(figsize=(9, 4))
    plt.hist(raw.ravel(), bins=120, alpha=0.55, label="сире спостереження")
    plt.hist(unconstrained.ravel(), bins=120, alpha=0.55, label="безумовний розв'язок")
    plt.axvline(0.0, color="red", ls="--", label="межі допустимої області")
    plt.axvline(1.0, color="red", ls="--")
    plt.xlabel("яскравість пікселя")
    plt.ylabel("кількість пікселів")
    plt.legend()
    plt.savefig("outputs/lab5_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    res_pg = projected_gradient(raw, lam_star, L, accelerated=False)
    res_fista = projected_gradient(raw, lam_star, L, accelerated=True)
    res_pen = penalty_method(raw, lam_star, L)

    print(f"\nпроєкція градієнта : ітер = {res_pg['iterations']}, "
          f"разом звернень = {res_pg['n_f'] + res_pg['n_grad']}, "
          f"E = {res_pg['E']:.8f}, порушення = {violation(res_pg['I'])['max']:.2e}")
    print(f"FISTA              : ітер = {res_fista['iterations']}, "
          f"разом звернень = {res_fista['n_f'] + res_fista['n_grad']}, "
          f"E = {res_fista['E']:.8f}, порушення = {violation(res_fista['I'])['max']:.2e}")
    v_pen = violation(res_pen["I"])
    print(f"квадратичний штраф : зовн. кроків = {len(res_pen['rows'])}, "
          f"разом звернень = {res_pen['n_f'] + res_pen['n_grad']}, "
          f"E = {res_pen['E']:.8f}, порушення = {v_pen['max']:.2e}, поза Ω = {v_pen['count']}")

    E5_star = min(res_pg["E"], res_fista["E"])
    plt.figure(figsize=(9, 5))
    for name, res in (("проєкція градієнта", res_pg), ("FISTA", res_fista)):
        ys = [max(r["E"] - E5_star, 1e-16) for r in res["rows"]]
        plt.semilogy(range(len(ys)), ys, label=name)
    plt.xlabel("номер ітерації")
    plt.ylabel("E(Ik) − E*")
    plt.title("частина 5: збіжність проєкційних методів")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("outputs/lab5_conv.png", dpi=150, bbox_inches="tight")
    plt.close()

    save_png(res_pg["I"], "outputs/lab5_restored.png")
    psnr_lab5 = psnr(res_pg["I"], clean)
    print(f"PSNR з обмеженнями (5) = {psnr_lab5:.4f} дБ")

    print("\n__обов'язкові експерименти (5)__")
    res_pg_wrong = projected_gradient(raw, lam_star, L, stop_by_grad=True)
    print(f"дослід 1 (зупинка за ‖∇E‖): ітер = {res_pg_wrong['iterations']}, "
          f"збігся = {res_pg_wrong['converged']}, кінцева ‖∇E‖ = {res_pg_wrong['grad']:.4f}, "
          f"‖Gh‖ = {res_pg_wrong['G']:.2e}")

    for g_factor, outer_lim in [(2.0, 22), (100.0, 5)]:
        rp_exp = penalty_method(raw, lam_star, L, growth=g_factor, outer=outer_lim)
        vp_exp = violation(rp_exp["I"])
        print(f"дослід 2 (множник росту β={g_factor:<5.1f}): зовн. ітер = {len(rp_exp['rows'])}, "
              f"разом звернень = {rp_exp['n_f'] + rp_exp['n_grad']}, порушення = {vp_exp['max']:.2e}")

    print("дослід 3 (ціна обумовленості Lr = L + 2r):")
    for row in res_pen["rows"]:
        print(f"  s={row['s']}: r={row['r']:<9.0f} Lr={row['L_r']:<12.4f} h={row['h']:.3e} "
              f"порушення={row['violation']:.2e} позаΩ={row['outside']}")

    # активна мапа обмежень
    active_map = np.full_like(res_pg["I"], 0.5)
    active_map[res_pg["I"] <= 1e-12] = 0.0
    active_map[res_pg["I"] >= 1.0 - 1e-12] = 1.0
    save_png(active_map, "outputs/lab5_active_map.png")
    n_low = int(np.count_nonzero(res_pg["I"] <= 1e-12))
    n_high = int(np.count_nonzero(res_pg["I"] >= 1.0 - 1e-12))
    print(f"дослід 4 (мапа активних обмежень збережена): на 0 стоять {n_low} пікс., на 1 стоять {n_high} пікс.")


if __name__ == "__main__":
    main()