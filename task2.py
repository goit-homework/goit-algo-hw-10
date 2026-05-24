"""
Завдання 2: інтеграл методом Монте-Карло.

Беремо ту саму функцію що в умові — f(x) = x^2 на [0, 2].
Аналітичний результат: x^3/3 від 0 до 2 = 8/3 ≈ 2.6667.

Реалізовано два варіанти MC:
  1) "дартс у коробку" (геометричний) — кидаємо точки у прямокутник,
     рахуємо частку що потрапили під криву.
  2) "середнє значення" (статистичний) — беремо середнє f(x) для
     випадкових x, множимо на ширину інтервалу.

Другий зазвичай точніший при однаковій кількості точок, перший —
зручніше візуалізувати.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi


def f(x):
    return x ** 2


A, B = 0.0, 2.0
ANALYTICAL = (B ** 3 - A ** 3) / 3


# ---------- два варіанти Монте-Карло ----------

def mc_darts(n: int, rng: np.random.Generator) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Метод "кидання дартсів". Випадково розкидаємо точки у прямокутнику
    [a, b] × [0, ymax], рахуємо частку що потрапили під криву.
    Інтеграл ≈ (площа прямокутника) × (частка точок під кривою).
    """
    ymax = f(B)  # бо f монотонно зростає на [0, 2]
    xs = rng.uniform(A, B, n)
    ys = rng.uniform(0, ymax, n)
    under = ys <= f(xs)
    box_area = (B - A) * ymax
    estimate = box_area * under.mean()
    return estimate, xs, ys, under


def mc_mean(n: int, rng: np.random.Generator) -> float:
    """
    Метод "середнього значення". Інтеграл = (b-a) × середнє(f(x))
    для x ~ Uniform(a, b).
    """
    xs = rng.uniform(A, B, n)
    return (B - A) * f(xs).mean()


# ---------- візуалізації ----------

def plot_darts(estimate: float, xs, ys, under, filename: str = "mc_darts.png") -> None:
    """Графік із завдання + накладені точки симуляції."""
    fig, ax = plt.subplots(figsize=(8, 6))

    grid_x = np.linspace(-0.5, 2.5, 400)
    ax.plot(grid_x, f(grid_x), "r", linewidth=2, label="f(x) = x²")

    fill_x = np.linspace(A, B, 200)
    ax.fill_between(fill_x, f(fill_x), color="gray", alpha=0.25, label="справжня площа")

    # випадкові точки
    ax.scatter(xs[under], ys[under], s=4, c="#43a047", alpha=0.5, label="під кривою")
    ax.scatter(xs[~under], ys[~under], s=4, c="#e53935", alpha=0.5, label="над кривою")

    ax.axvline(A, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(B, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, max(f(grid_x)) + 0.3])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(f"Монте-Карло (дартс): N={len(xs)},  оцінка ≈ {estimate:.4f}  (точно: {ANALYTICAL:.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  графік збережено: {filename}")


def plot_convergence(filename: str = "mc_convergence.png") -> None:
    """Як похибка падає з ростом N. Очікувано — як 1/√N."""
    rng = np.random.default_rng(42)
    sizes = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]

    # для кожного N усереднюємо по trials повторів
    trials = 30
    errors_darts = []
    errors_mean = []

    for n in sizes:
        errs_d, errs_m = [], []
        for _ in range(trials):
            est_d, *_ = mc_darts(n, rng)
            est_m = mc_mean(n, rng)
            errs_d.append(abs(est_d - ANALYTICAL))
            errs_m.append(abs(est_m - ANALYTICAL))
        errors_darts.append(np.mean(errs_d))
        errors_mean.append(np.mean(errs_m))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(sizes, errors_darts, "o-", label="дартс", color="#1976d2")
    ax.loglog(sizes, errors_mean, "s-", label="середнє значення", color="#388e3c")

    # лінія 1/√N для орієнтації
    ref = [errors_mean[0] * (sizes[0] / s) ** 0.5 for s in sizes]
    ax.loglog(sizes, ref, "--", color="gray", alpha=0.5, label="теоретично ~ 1/√N")

    ax.set_xlabel("N (кількість точок)")
    ax.set_ylabel("середня абс. похибка")
    ax.set_title(f"Збіжність Монте-Карло (середнє по {trials} запусках)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  графік збережено: {filename}")


# ---------- основний прогон ----------

def main() -> None:
    print("=" * 60)
    print(f"  Інтегруємо f(x) = x² на [{A}, {B}]")
    print("=" * 60)
    print(f"  аналітично:        {ANALYTICAL:.10f}  (8/3)")

    quad_result, quad_err = spi.quad(f, A, B)
    print(f"  scipy.integrate.quad: {quad_result:.10f}  (похибка ~ {quad_err:.2e})")
    print()

    rng = np.random.default_rng(42)

    print("  Монте-Карло, різні N (одиничний запуск):")
    print(f"  {'N':>9} {'darts':>15} {'mean':>15}  abs.err (mean)")
    print("  " + "-" * 56)
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        est_d, *_ = mc_darts(n, rng)
        est_m = mc_mean(n, rng)
        err = abs(est_m - ANALYTICAL)
        print(f"  {n:>9} {est_d:>15.6f} {est_m:>15.6f}  {err:.6f}")

    print()
    print("  Малюємо графіки...")
    final_n = 5_000
    est, xs, ys, under = mc_darts(final_n, rng)
    plot_darts(est, xs, ys, under)
    plot_convergence()


if __name__ == "__main__":
    main()
