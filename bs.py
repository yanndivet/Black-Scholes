import numpy as np
from scipy.stats import norm
from typing import Tuple
from math import pi

sqrt_2pi = np.sqrt(2*pi)
number_days_per_year = 252 # used for theta calculation
N = norm.cdf


def N_delta(x:np.array) -> np.array:
    return np.exp(-np.square(x)/2) / sqrt_2pi
    

def BS_opt(S:np.array, K:np.array, T: np.array, r: np.array, q: np.array, sigma: np.array, call_put: np.array) -> np.array:
    r_less_q_time_t = np.multiply(np.subtract(r,q), T)
    exp_r_less_q_time_t = np.exp(r_less_q_time_t)
    sigma_time_sqrt_t = np.multiply(sigma, np.sqrt(T))
    forward = np.multiply(S, exp_r_less_q_time_t)
    ln_f_over_k = np.log(np.divide(forward,K))   # same as ln(S/K) + (r-q)*T
    d1 = np.divide(ln_f_over_k, sigma_time_sqrt_t) + 0.5 * sigma_time_sqrt_t
    d2 = np.subtract(d1, sigma_time_sqrt_t)
    indicator = np.where(call_put=="P", -1, 1)
    d1_ = np.multiply(d1, indicator)
    d2_ = np.multiply(d2, indicator)
    
    return np.multiply(
                np.divide(indicator, exp_r_less_q_time_t), 
                np.subtract(
                    np.multiply(forward, N(d1_)),
                    np.multiply(K, N(d2_))
                           )
                        )

def BS_with_greeks(S:np.array, K:np.array, T: np.array, r: np.array, q: np.array, sigma: np.array, call_put: np.array) -> Tuple[np.array]:
    """
    

    Parameters
    ----------
    S : np.array
        Price of the underlying.
    K : np.array
        Exercice price of the option.
    T : np.array
        Time to maturity in year.
    r : np.array
        risk free interest rate in absolute: 5% interest rate should be entered as 0.05.
    q : np.array
        dividend yield in absolute: 3% dividend yield should be entered as 0.03 .
    sigma : np.array
        annualized volatility in absolute: 30% volatility should be entered as 0.3.
    call_put : np.array
        call or put option: 'C' for call, 'P' for put, case sensitive.

    Returns
    -------
    price : np.array
        price of the option.
    delta : np.array
        first derivative of the price with respect to the underlying price.
    gamma : np.array
        second derivative of the price with respect to the underlying price.
    theta : np.array
        first derivative of the price with respect to the time to maturity.
    vega : np.array
        first derivative of the price with respect to the underlying volatility.
    rho : np.array
        first derivative of the price with respect to the interest rate.

    """
    exp_r_T = np.exp(np.multiply(r,T))
    exp_minus_q_T = np.exp(-np.multiply(q,T))
    exp_r_less_q_time_t = np.multiply(exp_r_T, exp_minus_q_T)
    sigma_time_sqrt_t = np.multiply(sigma, np.sqrt(T))
    forward = np.multiply(S, exp_r_less_q_time_t)
    ln_f_over_k = np.log(np.divide(forward,K))   # same as ln(S/K) + (r-q)*T
    d1 = np.divide(ln_f_over_k, sigma_time_sqrt_t) + 0.5 * sigma_time_sqrt_t
    d2 = np.subtract(d1, sigma_time_sqrt_t)
    indicator = np.where(call_put=="P", -1, 1)
    d1_ = np.multiply(d1, indicator)
    d2_ = np.multiply(d2, indicator)
    n_d1_ = N(d1_)
    n_d2_ = N(d2_)
    exp_minus_qt_N_delta_d1 = np.multiply(N_delta(d1_),exp_minus_q_T)
    F_nd1 = np.multiply(forward, n_d1_)
    K_nd2 = np.multiply(K, n_d2_)
    price= np.multiply(
                np.divide(indicator, exp_r_T), 
                np.subtract(F_nd1,K_nd2)
                        )
    delta = np.multiply(indicator,exp_minus_q_T) * n_d1_
    gamma = np.divide(
                exp_minus_qt_N_delta_d1,
                np.multiply(S,sigma_time_sqrt_t)
                )
    vega = np.multiply(
                exp_minus_qt_N_delta_d1,
                np.multiply(S, np.sqrt(T))
        ) / 100
    rho = np.multiply(
            np.multiply(K_nd2, T),
            np.divide(indicator, exp_r_T)
            ) / 100
    theta_1 = - 0.5 * np.multiply(
                        exp_minus_qt_N_delta_d1,
                        np.divide( 
                            np.multiply(S, sigma),
                            np.sqrt(T)
                                )
                            )
    theta_2 =  (- np.multiply(r, K_nd2) + np.multiply(q, F_nd1)) / exp_r_T
                                    
    theta = np.add(     theta_1,
                        theta_2
                        ) / number_days_per_year
    return price, delta, gamma, theta, vega, rho


