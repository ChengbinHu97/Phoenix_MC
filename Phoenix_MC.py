import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


def Phoenix_MC(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon,Simulations):
    payoff = []
    knock_out_times = 0
    knock_in_times = 0
    N = int(360 * T)
    dt = T / N
    for i in range(Simulations):
        epsilon1 = np.random.normal(0, 1, N)
        stock_price = np.cumprod(np.exp((R - Q - 0.5 * Sigma ** 2) * dt + Sigma * epsilon1 * np.sqrt(dt))) * S
        obs = slice(30 - 1, N, 30)
        stock_price_obs = stock_price[obs]
        if stock_price_obs.max() >= U:

            knock_out_times += 1
            ix_to_ko = np.argmax(stock_price_obs >= U)
            ix_to_paid = np.argwhere(stock_price_obs > L_Coupon)
            ix_to_paid = ix_to_paid[ix_to_paid <= ix_to_ko]

            pv_payoff = np.sum((Coupon * S / 360 * 30) / (1 + R) ** (30 * (ix_to_paid + 1) / 360))
            payoff.append(pv_payoff)

        else:
            ki_indicator = min(stock_price) <= L
            knock_in_times += ki_indicator
            ix_to_paid = np.argwhere(stock_price_obs > L_Coupon)
            pv_payoff = ki_indicator * (min(stock_price[-1] - X_Put, 0) / ((1 + R) ** T)
                                        + np.sum((Coupon * S / 360 * 30) / (1 + R) ** ((30 * (ix_to_paid + 1)) / 360)))\
                        + (1 - ki_indicator) * np.sum((Coupon * S / 360 * 30) / (1 + R) ** (30 * (ix_to_paid + 1) / 360))
            payoff.append(pv_payoff)

    option_price = np.sum(payoff) / Simulations
    return option_price


def compute_greeks(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon, Simulations):
    epsilon = 0.01
    s_ini = S
    c = Coupon
    p_ini = Phoenix_MC(s_ini, X_Put, Sigma, R, Q, T, U, L, L_Coupon, c, Simulations)
    s1 = s_ini * (1 + epsilon)
    p1 = Phoenix_MC(s1, X_Put, Sigma, R, Q, T, U, L, L_Coupon, c, Simulations)
    s2 = s_ini * (1 - epsilon)
    p2 = Phoenix_MC(s2, X_Put, Sigma, R, Q, T, U, L, L_Coupon, c, Simulations)
    delta = (p1 - p_ini) / (epsilon * s_ini)
    gamma = (p1 + p2 - 2 * p_ini) / (epsilon ** 2 * s_ini ** 2)

    Sigma1 = Sigma + epsilon
    p4 = Phoenix_MC(s_ini, X_Put, Sigma1, R, Q, T, U, L, L_Coupon, c, Simulations)
    vega = (p4 - p_ini) / epsilon

    T1 = T - 1 / 365
    p5 = Phoenix_MC(s_ini, X_Put, Sigma, R, Q, T1, U, L, L_Coupon, c, Simulations)
    theta = (p5 - p_ini) / (- 1 / 365)
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
    Simulations = 1000000

    op = Phoenix_MC(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon, 1000000)
    result = op
    print("--------------------------------------")
    print("option_price =", result)
    print("--------------------------------------")
    d = compute_greeks(S, X_Put, Sigma, R, Q, T, U, L, L_Coupon, Coupon, Simulations)
    print("delta = ",  d[0])
    print("gamma = ", d[1])
    print("vega = ", d[2])
    print("theta = ", d[3] / 360)
    print("--------------------------------------")








