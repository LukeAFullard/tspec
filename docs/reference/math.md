# Mathematical Reference

This page maps every algorithm implemented in `tspec` directly to its peer-reviewed mathematical formulation.

---

## 1. Weighted Least-Squares Spectral Analysis (LSSA)

*Primary sources: Ghaderpour & Pagiatakis (2017); Ghaderpour et al. (2018); Ghaderpour (2021).*

Given unequally spaced observations $\{(t_i, f_i)\}_{i=1}^n$ with symmetric positive definite weight matrix $P \in \mathbb{R}^{n \times n}$ (scalar, 1D diagonal $P = \text{diag}(p_1, \dots, p_n)$, or full inverse covariance $P = C_f^{-1}$):

1. **Known Constituents:** Let $D \in \mathbb{R}^{n \times n_c}$ contain the trend basis (piecewise intercepts, piecewise or shared slopes, optional quadratic/cubic polynomials) and sinusoids of known cyclic frequencies. Factor $P = W^T W$ (via $\sqrt{p_i}$ for 1D weights or Cholesky $P = L L^T$, $W = L^T$ for 2D SPD weights) and solve via thin QR of $\tilde{D} = W D = Q R$:
   $$\hat{c} = R^{-1} Q^T \tilde{f}, \qquad \tilde{r} = \tilde{f} - Q (Q^T \tilde{f}), \qquad \|r\|_P^2 = \|\tilde{r}\|_2^2,$$
   $$\widehat{\text{Cov}}(\hat{c}) = \frac{\|r\|_P^2}{n - n_c} R^{-1} (R^{-1})^T.$$
2. **Vectorized Orthogonal Schur-Complement Spectrum:** For each candidate frequency $\omega_k \in \Omega$, let $\Phi_k = [\cos(2\pi \omega_k t_0),\, \sin(2\pi \omega_k t_0)]$ and project out $\text{col}(\tilde{D})$:
   $$\tilde{\Phi}_k^\perp = (I - Q Q^T) W \Phi_k.$$
   Then $S_k = (\tilde{\Phi}_k^\perp)^T \tilde{\Phi}_k^\perp = \Phi_k^T P \Phi_k - \Phi_k^T P D (D^T P D)^{-1} D^T P \Phi_k$ is the exact $2 \times 2$ Schur complement, and
   $$\hat{c}_k = S_k^{-1} (\tilde{\Phi}_k^\perp)^T \tilde{r}, \qquad s(\omega_k) = \frac{\tilde{r}^T \tilde{\Phi}_k^\perp \hat{c}_k}{\|\tilde{r}\|_2^2} \in [0, 1].$$
3. **Exact Null Distribution:** Under i.i.d. Gaussian noise (or correlated Gaussian noise with exact $P = C_f^{-1}$), $s(\omega_k) \sim \text{Beta}(1, \nu / 2)$ with $\nu = n - n_c - 2$, giving:
   $$s_{\text{crit}}(\alpha) = 1 - \alpha^{2/\nu}, \qquad p(\omega_k) = \bigl(1 - s(\omega_k)\bigr)^{\nu/2}.$$

---

## 2. Anti-Leakage LSSA (ALLSSA)

*Primary source: Ghaderpour, Liao, & Lamoureux (2018).*

ALLSSA iteratively extracts significant sinusoids simultaneously with the trend to prevent spectral leakage on irregular grids:
1. Evaluate LSS on coarse grid $\Omega$ conditional on current frequency set $F$.
2. Select $\omega^* = \arg\max_{\omega \in \Omega} s(\omega)$, round half-up to integer $\omega_0$, and temporarily remove the highest existing frequency in $F$ rounding to $\omega_0$.
3. Refine LSS on $[\omega_0 - 0.5, \omega_0 + 0.5]$ at resolution $10^{-d}$ (clamped strictly positive in default mode).
4. Accept $\omega_{\text{new}}$ if $s(\omega_{\text{new}}) > s_{\text{crit}}(\alpha)$; terminate when $F$ stabilizes (with explicit `max_iter` and cycle guards).

---

## 3. JUST Jump Detection & Decomposition

*Primary sources: Ghaderpour & Vujadinovic (2020a); Ghaderpour (2021).*

- **Windowed 2-Piece Scan:** Within each sliding window (count-based `size=3M, step=M` or calendar `window_duration`), scan $j \in \{3, \dots, n_w - 3\}$, fitting a discontinuous 2-piece linear trend jointly with ALLSSA/OLS harmonics, selecting $j^*$ minimizing $\|r\|_P^2$:
  $$\text{Mag} = (\hat{b}_2 - \hat{b}_1) + (\hat{a}_2 - \hat{a}_1) t_{0,j^*}, \qquad \text{Dir} = \hat{a}_2 - \hat{a}_1.$$
- **Nested Partial $F$-Tests (clean-room additions):** Selecting harmonics under the **no-jump null** model first (for `season="allssa"`, preventing harmonic-mining inflation under $H_0$) and holding those harmonics fixed across nested models $D_{\text{null}} \subset D_{\text{hinge}} \subset D_{\text{full}}$, `tspec` computes:
  1. `jump_pvalue` (2 df): discontinuous 2-piece linear ($\text{RSS}_1$) vs. continuous 1-piece linear ($\text{RSS}_0$), $F_{\text{jump}} = \frac{(\text{RSS}_0 - \text{RSS}_1)/2}{\text{RSS}_1 / (n_w - n_c)}$.
  2. `step_pvalue` (1 df): discontinuous 2-piece linear ($\text{RSS}_1$) vs. optimal continuous hinge across the window ($\text{RSS}_{\text{hinge}} = \min_k \text{RSS}_{\text{hinge}}(k)$), $F_{\text{step}} = \frac{\text{RSS}_{\text{hinge}} - \text{RSS}_1}{\text{RSS}_1 / (n_w - n_c)}$.
  Optional `bonferroni_window=True` scales screening $p$-values by $\max(1, n_w - 5)$ to account for within-window split selection.
- **Decomposition:** Overlapping windows are combined by Gaussian-weighted averaging with weights $w(t) = \exp\!\left(\frac{\ln 0.1}{(W/6)^2}(t - \tau_0)^2\right)$ (using floating-point sampling rate in default mode so arbitrary time units such as days or seconds never underflow), enforcing zero mean on each local seasonal component before blending.

---

## 4. Near-Real-Time Monitoring (`JUSTmonitor`, v1.1)

*Primary sources: Ghaderpour & Vujadinovic (2020b); Lan & DeMets (1983).*

1. **Historical Model & Causal Preprocessing:** Fit a linear trend plus seasonal harmonics (`ALLSSA`, `OLS`, or `none`) on the stable historical segment $t[\text{history}:\text{start}]$ and evaluate the forecast $\hat{f}(t)$ forward through the monitoring period, yielding forecast residuals $e(t) = f(t) - \hat{f}(t)$. When censored data are present, `ndimpute` is fitted strictly causally on $t[\text{history}:\text{start}]$ with monitoring observations imputed sequentially without future look-ahead.
2. **Predictive Residual Covariance (`account_for_forecast_uncertainty=True`):** Under independent Gaussian noise, historical OLS orthogonality implies $\text{Cov}(e_{\text{in}}, e_{\text{out}}) = \mathbf{0}$, while out-of-sample forecast residuals have exact covariance
   $$\Sigma_{\text{out}} = \sigma^2 \Bigl(P_{\text{out}}^{-1} + D_{\text{out}} (D_{\text{his}}^T P_{\text{his}} D_{\text{his}})^{-1} D_{\text{out}}^T\Bigr).$$
   In default mode, `tspec` applies block-diagonal GLS weights $P_{\text{win}} = \text{block\_diag}(P_{\text{in}}, \Sigma_{\text{out}}^{-1})$ at each monitoring step so finite-sample historical slope/seasonal estimation uncertainty does not cause false alarms on long horizons (`parity_mode=True` disables this for exact upstream matching).
3. **Sequential Residual LSSA:** At each incoming observation $d = 1, \dots, D$, compute the constant-detrended normalized LSS of $e$ over the rolling window (expanding from $L_h$ or constant-width sliding) across $K = 35$ frequencies $\Omega_{\text{mon}} = \{0.1, 0.2, \dots, 3.5\}$ cy/yr:
   - Low-frequency peak ($\max_{\omega < 1} s(\omega) > s_{\text{crit}}$) $\implies$ **abrupt disturbance** (`type=1`).
   - High-frequency peak exclusively ($\max_{\omega \ge 1} s(\omega) > s_{\text{crit}}$) $\implies$ **phenological disturbance** (`type=2`).
4. **Multiplicity & Sequential-Peeking Control:**
   - **Per-step across-time budget $\alpha_d$** over planned horizon $H$:
     - `pocock`: $\alpha_d = 1 - (1 - \alpha)^{1/H}$ (constant Šidák allocation).
     - `obrien_fleming`: Lan–DeMets incremental alpha-spending $\alpha_d = \alpha^*(d/H) - \alpha^*((d-1)/H)$ where $\alpha^*(u) = 2\bigl(1 - \Phi(z_{\alpha/2}/\sqrt{u})\bigr)$; steps past $H$ emit `HORIZON_EXCEEDED` and retain the step-$H$ floor.
     - `consecutive_k`: $\alpha_d = \alpha$, requiring $k$ consecutive exceedances (open-ended streams).
   - **Per-step multi-frequency adjustment** across $K$ scanned frequencies: Šidák $\alpha_{\text{adj}} = 1 - (1 - \alpha_d)^{1/K}$ or Bonferroni $\alpha_{\text{adj}} = \alpha_d / K$, applied to the exact Beta threshold $s_{\text{crit}}(\alpha_{\text{adj}}) = 1 - \alpha_{\text{adj}}^{2/\nu}$.

---

## 5. STPD & Piecewise Linear Trend Estimation (TPTR)

*Primary source: Ghaderpour et al. (2024).*

- **Asymmetric Sequential Two-Step Continuous OLS:** For each interior split $j \in [\text{margin}, n_w - \text{margin})$ ($\text{margin} \ge 2$):
  1. Fit unconstrained 2-parameter OLS on $i \in \{0, \dots, j\}$: $\hat{y}_1(t) = \hat{b}_1 + \hat{a}_1 t$.
  2. Anchor the second segment at $(t_j, \hat{b}_1 + \hat{a}_1 t_j)$ and fit 1-parameter slope $\hat{a}_2$ over $i \in \{j, \dots, n_w - 1\}$.
  3. Repeat in reversed time $t' = t_{n_w - 1} - t$; select the direction with smaller $\|r\|_P$ and record `fb_diff` $= |\tau_f - \tau_b|$.
- **Significance, SNR & NDRI:**
  - **Default exact fixed-knot Student $t$-test ($df = n_w - 3$):** Because the second segment is anchored at $\hat{y}_1(t_j)$, $\hat{a}_1$ and $\hat{a}_2$ are negatively correlated ($\text{Cov}(\hat{a}_1, \hat{a}_2) < 0$). Writing $\hat{a}_1 - \hat{a}_2 = \mathbf{v}^T \mathbf{y}$ in closed $O(n_w)$ form yields the exact variance $\widehat{\text{Var}}(\hat{a}_1 - \hat{a}_2) = \frac{\text{RSS}}{n_w - 3} \sum_{i=1}^{n_w} v_i^2 / w_i$.
  - **`parity_mode=True` approximation ($df = n_w - 2$):** Uses Ghaderpour et al. (2024)'s independent-segment approximation $t = |\hat{a}_1 - \hat{a}_2| / \sqrt{\text{SE}_1^2 + \text{SE}_2^2}$.
  - Weighted $\text{SNR}_{\text{raw}} = \|\hat{y}\|_P / \|f - \hat{y}\|_P$ and shift-invariant $\text{SNR}_{\text{centered}} = \|\hat{y} - \bar{\hat{y}}_P\|_P / \|f - \hat{y}\|_P$.
  - Weighted $\text{NDRI}_{L_2} = \frac{|\|r_1\|_P - \|r_2\|_P|}{\|r_1\|_P + \|r_2\|_P}$ and density-invariant $\text{NDRI}_{\text{RMSE}} = \frac{|\text{RMSE}_1 - \text{RMSE}_2|}{\text{RMSE}_1 + \text{RMSE}_2}$.

---

## 6. Censored Imputation (`ndimpute`)

*Primary sources: Helsel (2012); Turnbull (1976); Fullard (2026).*

- **Left-censored ROS:** Hirsch–Stedinger / Helsel–Cohn plotting positions with OLS probability-plot regression (`NADA::ros`-exact). `tspec` permutes tied detection-limit groups in `'quantile'` mode and defaults to `'stochastic'` to eliminate chronological rank-order trend injection.
- **Parametric MLE:** Full censored likelihood $L(\theta) = \prod_{\text{obs}} f(x_i) \prod_{\text{left}} F(L_i) \prod_{\text{right}} S(R_i) \prod_{\text{int}} [F(r_i) - F(l_i)]$ (`survival::survreg`-exact) with numerically stable truncated expectations and draws.
