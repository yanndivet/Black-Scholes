# Black-Scholes
The [Black-Scholes formula](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) (aka Black-Scholes-Merton formula) enables the pricing of [European styled](https://en.wikipedia.org/wiki/Option_style) [options](https://en.wikipedia.org/wiki/Option_(finance)) under various (simplifying and sometimes questionable) assumptions.
The main idea behind the BS (BSM) model leading to the close form formula is that an option can be dynamically replicated by a portfolio consisting of the underlying and a [zero coupon bond](https://en.wikipedia.org/wiki/Zero-coupon_bond). In particular the risk or uncertainty of the option, which comes from uncertainty of the underlying, can be locally offset by trading the right amount of the underlying. 


The python code is vectorized using numpy to improve the performance, in particular to perform a risk analysis along a vector of values for one of the parameters at a time. Closed form formula are available for the option [greeks](https://en.wikipedia.org/wiki/Greeks_(finance)) which simplify the computation of sensitivities, without relying on the more general finite difference method.
