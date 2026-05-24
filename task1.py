"""
Завдання 1: розмін монет двома способами — жадібним і динамічним.

Канонічний український набір монет [50, 25, 10, 5, 2, 1] є таким, що
жадібний алгоритм завжди дає оптимум. Але це не означає що жадібний
у загальному випадку правильний — ми це теж демонструємо нижче.
"""

import time
from typing import Iterable


UKR_COINS = (50, 25, 10, 5, 2, 1)


def find_coins_greedy(amount: int, coins: Iterable[int] = UKR_COINS) -> dict[int, int]:
    """
    Жадібний підхід: на кожному кроці беремо найбільший номінал що
    помістився. Працює гарантовано тільки для так званих "канонічних"
    наборів монет (укр/європейський/доларовий — канонічні).

    Складність: O(k), де k — кількість номіналів.
    """
    if amount < 0:
        raise ValueError("сума не може бути від'ємною")

    result: dict[int, int] = {}
    remaining = amount
    for coin in sorted(coins, reverse=True):
        if remaining == 0:
            break
        if coin <= remaining:
            count, remaining = divmod(remaining, coin)
            result[coin] = count
    return result


def find_min_coins(amount: int, coins: Iterable[int] = UKR_COINS) -> dict[int, int]:
    """
    Динамічне програмування: для кожної суми від 0 до amount рахуємо
    мінімальну кількість монет. Окремо тримаємо який номінал ми обрали
    останнім на цьому етапі — це дозволяє відновити набір монет.

    Складність: O(amount × k) часу і O(amount) пам'яті.
    Гарантовано оптимальний результат для будь-якого набору монет.
    """
    if amount < 0:
        raise ValueError("сума не може бути від'ємною")
    if amount == 0:
        return {}

    INF = float("inf")
    # min_count[s] — мінімальна кількість монет щоб набрати суму s
    min_count = [0] + [INF] * amount
    # last_coin[s] — номінал, який ми використали останнім щоб досягти s
    last_coin = [0] * (amount + 1)

    coin_list = list(coins)
    for s in range(1, amount + 1):
        for c in coin_list:
            if c <= s and min_count[s - c] + 1 < min_count[s]:
                min_count[s] = min_count[s - c] + 1
                last_coin[s] = c

    if min_count[amount] == INF:
        raise ValueError(f"неможливо набрати суму {amount} з монет {coin_list}")

    # відновлюємо словник монет, рухаючись від amount назад
    result: dict[int, int] = {}
    s = amount
    while s > 0:
        c = last_coin[s]
        result[c] = result.get(c, 0) + 1
        s -= c

    # сортуємо за номіналом за зростанням — відповідає прикладу в умові
    return dict(sorted(result.items()))


def total_coins(coin_dict: dict[int, int]) -> int:
    return sum(coin_dict.values())


# ---------- демонстрація і бенчмарк ----------

def demo_basic() -> None:
    print("=" * 60)
    print("  Базова демонстрація на укр-наборі", UKR_COINS)
    print("=" * 60)
    for amount in [113, 999, 7, 1, 100]:
        greedy = find_coins_greedy(amount)
        dp = find_min_coins(amount)
        print(f"  сума {amount:>5}:")
        print(f"    greedy: {greedy}  (всього {total_coins(greedy)})")
        print(f"    dp:     {dp}  (всього {total_coins(dp)})")


def demo_greedy_fails() -> None:
    """Показуємо випадок коли жадібний дає НЕ оптимальний результат."""
    print()
    print("=" * 60)
    print("  Коли жадібний помиляється: набір [1, 3, 4], сума 6")
    print("=" * 60)
    bad_coins = (1, 3, 4)
    amount = 6

    greedy = find_coins_greedy(amount, bad_coins)
    dp = find_min_coins(amount, bad_coins)

    print(f"  greedy: {greedy}  -> {total_coins(greedy)} монети")
    print(f"  dp:     {dp}  -> {total_coins(dp)} монети  <-- справжній оптимум")
    print()
    print("  Жадібний бере 4+1+1, бо найбільший номінал — 4.")
    print("  DP знаходить 3+3, що краще. Для канонічних наборів (як укр)")
    print("  така ситуація неможлива, але для довільних — цілком.")


def benchmark() -> None:
    print()
    print("=" * 60)
    print("  Бенчмарк часу на укр-наборі")
    print("=" * 60)
    print(f"  {'amount':>10} {'greedy':>14} {'dp':>14}  speedup")
    print("  " + "-" * 56)

    for amount in [100, 1_000, 10_000, 100_000, 1_000_000]:
        t0 = time.perf_counter()
        for _ in range(100):
            find_coins_greedy(amount)
        t_greedy = (time.perf_counter() - t0) / 100

        # DP стає повільним на 1M, тому міряємо менше повторів
        n_dp_repeats = 1 if amount >= 100_000 else 5
        t0 = time.perf_counter()
        for _ in range(n_dp_repeats):
            find_min_coins(amount)
        t_dp = (time.perf_counter() - t0) / n_dp_repeats

        speedup = t_dp / t_greedy if t_greedy > 0 else float("inf")
        print(f"  {amount:>10} {t_greedy * 1e6:>12.2f}us {t_dp * 1000:>12.2f}ms  ~{speedup:>10.0f}x")


if __name__ == "__main__":
    demo_basic()
    demo_greedy_fails()
    benchmark()
