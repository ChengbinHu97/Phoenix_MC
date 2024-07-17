import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def up_and_out_cash_or_nothing_call(S, X, Sigma, R, Q, T, K, H):
    eta = -1
    phi = 1
    b = R - Q
    mu = (b - Sigma ** 2 / 2) / (Sigma ** 2)
    x1 = np.log(S / X) / (Sigma * np.sqrt(T)) + (mu + 1) * Sigma * np.sqrt(T)
    x2 = np.log(S / H) / (Sigma * np.sqrt(T)) + (mu + 1) * Sigma * np.sqrt(T)
    y1 = np.log(H ** 2 / (S * X)) / (Sigma * np.sqrt(T)) + (mu + 1) * Sigma * np.sqrt(T)
    y2 = np.log(H / S) / (Sigma * np.sqrt(T)) + (mu + 1) * Sigma * np.sqrt(T)
    B1 = K * np.exp(-R * T) * norm.cdf(phi * x1 - phi * Sigma * np.sqrt(T))
    B2 = K * np.exp(-R * T) * norm.cdf(phi * x2 - phi * Sigma * np.sqrt(T))
    B3 = K * np.exp(-R * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y1 - eta * Sigma * np.sqrt(T))
    B4 = K * np.exp(-R * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y2 - eta * Sigma * np.sqrt(T))
    if X > H:
        option_price = 0
    else:
        option_price = B1 - B2 + B3 - B4
    return option_price


def up_and_in_cash_or_nothing(S, Sigma, R, Q, T, K, H):
    eta = -1
    b = R - Q
    mu = (b - Sigma ** 2 / 2) / (Sigma ** 2)
    lambd = np.sqrt(mu ** 2 + 2 * R / Sigma ** 2)
    zeta = np.log(H / S) / (Sigma * np.sqrt(T)) + lambd * Sigma * np.sqrt(T)
    A5 = K * ((H / S) ** (mu + lambd) * norm.cdf(eta * zeta)
              + (H / S) ** (mu - lambd) * norm.cdf(eta * zeta - 2 * eta * lambd * Sigma * np.sqrt(T)))
    option_price = A5
    return option_price


def up_and_out_down_and_out_put(S, X, Sigma, R, Q, T, U, L):
    epsilon1 = 0
    epsilon2 = 0
    b = R - Q
    E = L * np.exp(epsilon2 * T)
    sum1 = 0
    sum2 = 0
    for n in range(-5, 6):
        mu1 = 2 * (b - epsilon2 - n * (epsilon1 - epsilon2)) / Sigma ** 2 + 1
        mu2 = 2 * n * (epsilon1 - epsilon2) / Sigma ** 2
        mu3 = 2 * (b - epsilon2 + n * (epsilon1 - epsilon2)) / Sigma ** 2 + 1
        y1 = (np.log(S * U ** (2 * n) / (E * L ** (2 * n))) + (b + Sigma ** 2 / 2) * T) / (Sigma * np.sqrt(T))
        y2 = (np.log(S * U ** (2 * n) / (X * L ** (2 * n))) + (b + Sigma ** 2 / 2) * T) / (Sigma * np.sqrt(T))
        y3 = (np.log(L ** (2 * n + 2) / (E * S * U ** (2 * n))) + (b + Sigma ** 2 / 2) * T) / (Sigma * np.sqrt(T))
        y4 = (np.log(L ** (2 * n + 2) / (X * S * U ** (2 * n))) + (b + Sigma ** 2 / 2) * T) / (Sigma * np.sqrt(T))

        sum1 += ((U ** n / L ** n) ** mu1 * (L / S) ** mu2
                 * (norm.cdf(y1) - norm.cdf(y2))
                 - (L ** (n + 1) / (U ** n * S)) ** mu3
                 * (norm.cdf(y3) - norm.cdf(y4)))

        sum2 += ((U ** n / L ** n) ** (mu1 - 2) * (L / S) ** mu2 *
                 (norm.cdf(y1 - Sigma * np.sqrt(T)) - norm.cdf(y2 - Sigma * np.sqrt(T)))
                 - (L ** (n + 1) / (U ** n * S)) ** (mu3 - 2)
                 * (norm.cdf(y3 - Sigma * np.sqrt(T)) - norm.cdf(y4 - Sigma * np.sqrt(T))))
    if S < L:
        option_price = 0
    else:
        option_price = - S * np.exp((b - R) * T) * sum1 + X * np.exp(-R * T) * sum2
    return option_price


def up_and_out_put(S, X, Sigma, R, Q, T, H):
    etaa = -1
    phii = -1
    b = R - Q
    mu = (b - Sigma ** 2 / 2) / Sigma ** 2
    x1 = np.log(S / X) / (Sigma * np.sqrt(T)) + (1 + mu) * Sigma * np.sqrt(T)
    x2 = np.log(S / H) / (Sigma * np.sqrt(T)) + (1 + mu) * Sigma * np.sqrt(T)
    y1 = np.log(H ** 2 / (S * X)) / (Sigma * np.sqrt(T)) + (1 + mu) * Sigma * np.sqrt(T)
    y2 = np.log(H / S) / (Sigma * np.sqrt(T)) + (1 + mu) * Sigma * np.sqrt(T)
    A = phii * S * np.exp((b - R) * T) * norm.cdf(phii * x1) \
        - phii * X * np.exp(-R * T) * norm.cdf(phii * x1 - phii * Sigma * np.sqrt(T))
    B = phii * S * np.exp((b - R) * T) * norm.cdf(phii * x2) \
        - phii * X * np.exp(-R * T) * norm.cdf(phii * x2 - phii * Sigma * np.sqrt(T))
    C = phii * S * np.exp((b - R) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(etaa * y1) \
        - phii * X * np.exp(-R * T) * (H / S) ** (2 * mu) * norm.cdf(etaa * y1 - etaa * Sigma * np.sqrt(T))
    D = phii * S * np.exp((b - R) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(etaa * y2) \
        - phii * X * np.exp(-R * T) * (H / S) ** (2 * mu) * norm.cdf(etaa * y2 - etaa * Sigma * np.sqrt(T))

    if X >= H:
        puo = B - D
    else:
        puo = A - C
    return puo


def phoenix(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon):
    s0 = S
    K = Coupon / 360 * 30 * s0

    High = U * np.exp(0.5826 * Sigma * np.sqrt(1 / 12))  #
    v1 = up_and_in_cash_or_nothing(S, Sigma, R, Q, T, K, High)  # OTU or American Digits

    v2 = 0
    for t in range(1, int(12 * T) + 1):

        Hit = U * np.exp(0.5826 * Sigma * np.sqrt(t / 12 / t))  #
        v2 += up_and_out_cash_or_nothing_call(S, L_Coupon, Sigma, R, Q, t / 12, K, Hit)

    upper = U * np.exp(0.5826 * Sigma * np.sqrt(1 / 12))
    # upper = U
    v3 = up_and_out_put(S, X_Put, Sigma, R, Q, T, upper)

    if S < L:
        v4 = 0
    else:
        # lower = L
        upper = U * np.exp(0.5826 * Sigma * np.sqrt(1 / 12))
        lower = L * np.exp(-0.5826 * Sigma * np.sqrt(1 / 365))
        v4 = up_and_out_down_and_out_put(S, X_Put, Sigma, R, Q, T, upper, lower)

    option_price = v1 + v2 - (v3 - v4)
    return option_price


def compute_greeks(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon):
    epsilon = 0.01
    s_ini = S
    c = Coupon
    p_ini = phoenix(s_ini, X_Put, Sigma, R, Q, T, U, L, L_Coupon, c)
    s1 = s_ini * (1 + epsilon)
    p1 = phoenix(s1, X_Put, Sigma, R, Q, T, U, L, L_Coupon, c)
    s2 = s_ini * (1 - epsilon)
    p2 = phoenix(s2, X_Put, Sigma, R, Q, T, U, L, L_Coupon, c)
    delta = (p1 - p_ini) / (epsilon * s_ini)
    gamma = (p1 + p2 - 2 * p_ini) / (epsilon ** 2 * s_ini ** 2)
    Sigma1 = Sigma + epsilon
    p4 = phoenix(s_ini, X_Put, Sigma1, R, Q, T, U, L, L_Coupon, c)
    vega = (p4 - p_ini) / epsilon

    T1 = T - 1 / 360
    p5 = phoenix(s_ini, X_Put, Sigma, R, Q, T1, U, L, L_Coupon, c)
    theta = (p5 - p_ini) / (- 1 / 360)
    return delta, gamma, vega, theta


if __name__ == '__main__':
    S = 100
    X_Put = 100
    Sigma = 0.18
    R = 0.03
    Q = 0.035
    T = 12 / 12
    U = 103
    L = 75
    L_Coupon = 75
    Coupon = 7 / 100

    op = phoenix(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon)
    result = op
    print("--------------------------------------")
    print("option_price =", result)
    print("--------------------------------------")
    d = compute_greeks(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon)
    print("delta = ", d[0])
    print("gamma = ", d[1])
    print("vega = ", d[2])
    print("theta = ", d[3])
    print("--------------------------------------")

