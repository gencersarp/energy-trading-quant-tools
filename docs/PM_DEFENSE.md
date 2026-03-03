# Energy Quant: Institutional PM Defense

## 1. Capital Allocation Upgrade
**Problem:** Sizing positions linearly in energy markets leads to ruin due to heavy tails.
**Solution:** Regime-Conditional Kelly & Volatility Targeting.

Given a belief state vector from our HMM $\pi_t = [P(Base), P(Spike)]$, the optimal fraction $f_t^*$ maximizes expected log-growth:
$$ \max_f \mathbb{E} [\log(1 + f \cdot r_{t+1}) | \mathcal{F}_t] $$

Using a Taylor expansion around $f=0$, the state-conditioned Kelly is:
$$ f_t^* \approx \frac{\sum \pi_i \mu_i}{\sum \pi_i (\sigma_i^2 + \mu_i^2) - (\sum \pi_i \mu_i)^2} $$
**Why it works:** In a spike regime, $\sigma_{spike}^2$ explodes, drastically reducing $f_t^*$ and auto-deleveraging the portfolio exactly when variance drag would destroy geometric compounding.

---

## 2. Storage Arbitrage Module (Battery LP vs SDDP)
**The Toy Model:** 
A basic LP optimizes:
$$ \max_{c_t, d_t} \sum_{t=0}^{T} [P_t d_t - P_t c_t - \kappa (c_t + d_t)] $$
Subject to: $SoC_{t+1} = SoC_t + \eta c_t - \frac{1}{\eta} d_t$

**The Reality (What you say in the interview):**
A deterministic LP requires *perfect foresight* of $P_t$, introducing immense lookahead bias. In reality, storage optimization is a sequential decision-making process under uncertainty. 
*Production implementation requires **Stochastic Dual Dynamic Programming (SDDP)** or **Model Predictive Control (MPC)**.* We must optimize the expected value over price scenario trees, continuously re-solving at $t$ using the latest forecast to dispatch in real-time. Furthermore, we must add piecewise linear bounds for non-linear cycle degradation (Depth of Discharge).

---

## 3. Tail Risk: Extreme Value Theory (EVT)
**Why Sharpe Fails:** The Sharpe ratio assumes symmetry and light tails. Power markets have massive positive kurtosis and extreme skew. Sharpe penalizes upside spikes as "variance."

**The Upgrade (POT Model):**
We apply the Pickands-Balkema-de Haan theorem. We take a high threshold $u$ (e.g., 95th percentile loss) and fit the exceedances to a Generalized Pareto Distribution (GPD).
$$ VaR_q = u + \frac{\beta}{\xi} \left( \left(\frac{N}{N_u}(1-q)\right)^{-\xi} - 1 \right) $$
$$ CVaR_q = \frac{VaR_q + \beta - \xi u}{1 - \xi} $$
*Winter Storm Uri Shock:* We apply a $10\sigma$ shift to the expected value using the heavy tail index $\xi$ to model system-wide correlated defaults.

---

## 4. Model Benchmarking
How does HMM compare?
1. **Markov-Switching AR (MS-AR):** Adds autoregressive dynamics within the states, capturing mean-reversion *inside* the regime rather than assuming independent draws.
2. **Jump-Diffusion (Merton):** Continuous time. Jumps arrive via Poisson process. Pro: Easy to price derivatives. Con: Misses the "duration" of a spike (jumps are instantaneous, HMM states persist).
3. **Hawkes Processes:** Jumps trigger more jumps (self-exciting). Highly accurate for modeling cascading liquidations / short squeezes in intraday power markets.

---

## 5. The Top 5 PM Interview Questions (and Answers)

**Q1: Your HMM transition matrix is static. Why is this a terminal flaw in power markets?**
*Answer:* Because power spikes are fundamentally driven by physical reality (weather, outages). Transition probability $P_{base \to spike}$ should not be constant; it should be an explicit function of exogenous variables $X_t$ (like Heating Degree Days or wind forecast errors). We need a Time-Varying Transition Probability (TVTP) HMM.

**Q2: How do you avoid lookahead bias when training an HMM historically?**
*Answer:* HMMs suffer from label switching and Baum-Welch local optima during rolling fits. If state 0 is "base" today, it might become "spike" tomorrow. To avoid leakage and maintain stability out-of-sample, we must use anchored warm-starts (initializing the new day's transition matrix with yesterday's optimized parameters) and explicitly enforce $\mu_0 < \mu_1$ during optimization bounds.

**Q3: You're using Kelly. Given the EVT tails in power, won't full Kelly bankrupt us in one gap move?**
*Answer:* Yes. Continuous-time Kelly assumes we can rebalance before hitting zero. In power, gap risk (discontinuous jumps) violates this. We use Fractional Kelly scaled by EVT CVaR (replacing variance in the denominator with our Expected Shortfall metric) to bound the probability of ruin to $< 1\%$.

**Q4: Transaction costs are modeled as bid-ask slippage. What physical constraints are missing?**
*Answer:* Imbalance penalties and margin. If we hold physical power into delivery and our load varies, the ISO penalizes us at punitive imbalance prices. Furthermore, exchanges require initial margin proportional to portfolio VaR; our strategy might look profitable but exceed capital constraints and face liquidation during a high-vol regime.

**Q5: What is the most obvious way your backtest overstates Alpha?**
*Answer:* Assuming we can actually fill the volume at the printed mid-price during a volatility breakout. In power markets, liquidity vanishes right before a spike. Our order will suffer massive market impact, or we simply won't get filled until the price has already gapped.