# Mathematical Reference

This page maps every algorithm implemented in `tspec` directly to its peer-reviewed mathematical formulation.

---

## 1. Weighted Least-Squares Spectral Analysis (LSSA)

*Primary sources: Ghaderpour & Pagiatakis (2017); Ghaderpour et al. (2018); Ghaderpour (2021).*

Given unequally spaced observations $\{(t_i, f_i)\}_{i=1}^n$ with symmetric positive definite weight matrix $P \in \mathbb{R}^{n \times n}$ (scalar, 1D diagonal $P = \text{diag}(p_1, \dots, p_n)$, or full inverse covariance $P = C_f^{-1}$):

1. **Known Constituents:** Let $D \in \mathbb{R}^{n \times n_c}$ contain the trend basis (piecewise intercepts, piecewise or shared slopes, optional quadratic/cubic polynomials) and sinusoids of known cyclic frequencies. Factor $P = W^T W$ (via $\sqrt{p_i}$ for 1D weights or Cholesky $P = L L^T$, $W = L^T$ for 2D SPD weights) and solve via thin QR of $\tilde{D} = W D = Q R$:
   $$\hat{c} = R^{-1} Q^T \tilde{f}, \qquad \tilde{r} = \tilde{f} - Q (Q^T \tilde{f}), \qquad \|r\|_P^2 = \|\tilde{r}\|_2^2,$$
   $$\widehat{\text{Cov}}(\hat{c}) = \frac{\|r\|_P^2}{n - n_c} R^{-1} (R^{-1})^T.$$
   Condition numbers are computed from exact singular values of $\tilde{D}$.
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
4. Accept $\omega_{\text{new}}$ only if it is significant **after accounting for the frequency search**; terminate when $F$ stabilizes (with explicit `max_iter` and cycle guards). `significance` selects the rule:
   - `"montecarlo"` *(default)*: under Gaussian white noise the normalized LSS is invariant to the error scale and the known-constituent coefficients, so drawing $B$ (= `n_mc`) standard-normal vectors in whitened coordinates, projecting off $\text{col}(WD)$ and taking $\max_{\omega \in G} s(\omega)$ over a dense grid $G$ covering every frequency the search can visit gives the exact null distribution of the searched maximum. Accept if $p_{\text{FW}} = (1 + \#\{\max^{(b)} \ge s(\omega_{\text{new}})\})/(B + 1) \le \alpha$.
   - `"sidak"`: accept if $s(\omega_{\text{new}}) > s_{\text{crit}}(\alpha_K)$ with $\alpha_K = 1 - (1 - \alpha)^{1/K_{\text{eff}}}$ and $K_{\text{eff}} = \min\!\bigl(|G|,\, \max(1, \lceil 3\, T \cdot \text{bandwidth}\rceil)\bigr)$ (oversampled periodogram effective trials, Horne & Baliunas 1986; Baluev 2008).
   - `"per_frequency"` (published rule; forced by `parity_mode=True`): $s(\omega_{\text{new}}) > s_{\text{crit}}(\alpha)$. This ignores the search and selects spurious cycles in roughly 20% of pure-noise series at $\alpha = 0.01$.

`lssa()` reports `peak_fap`, the family-wise false-alarm probability of the highest peak over the supplied grid: with `peak_fap="sidak"` (default), $p_{\text{FW}} = 1 - (1 - p_{\min})^{K_{\text{eff}}}$ where $K_{\text{eff}} = \min(|\Omega|, \max(1, \lceil 3\, T (\max\Omega - \min\Omega)\rceil))$; with `peak_fap="montecarlo"`, exact Monte Carlo draws of $\max_{\omega \in \Omega} s(\omega)$ are used. Per-frequency `pvalues` are unchanged.

---

## 3. JUST Jump Detection & Decomposition

*Primary sources: Ghaderpour & Vujadinovic (2020a); Ghaderpour (2021).*

- **Windowed 2-Piece Scan:** Within each sliding window (count-based `size=3M, step=M` or calendar `window_duration`), scan $j \in \{\text{min\_seg}, \dots, n_w - \text{min\_seg}\}$ (with $\text{min\_seg} = 3$ for non-robust fits and $\text{min\_seg} = \max(3, \min(5, \lfloor n_w / 5\rfloor))$ for robust fits), fitting a discontinuous 2-piece linear trend jointly with ALLSSA/OLS harmonics, selecting $j^*$ minimizing $\|r\|_P^2$ (or shared-scale Huber/Tukey loss for robust fits):
  $$\text{Mag} = (\hat{b}_2 - \hat{b}_1) + (\hat{a}_2 - \hat{a}_1) t_{0,j^*}, \qquad \text{Dir} = \hat{a}_2 - \hat{a}_1.$$
- **Nested Partial $F$-Tests (clean-room additions):** Selecting harmonics under the **no-jump null** model first (for `season="allssa"`, flooring candidate frequencies at $\max(\min\Omega, 0.85)$ when $\max\Omega \ge 1$ so sub-annual half-waves cannot absorb step jumps, and preventing harmonic-mining inflation under $H_0$) and holding those harmonics fixed across nested models $D_{\text{null}} \subset D_{\text{hinge}} \subset D_{\text{full}}$, `tspec` computes:
  1. `jump_pvalue` (2 df): discontinuous 2-piece linear ($\text{RSS}_1$) vs. continuous 1-piece linear ($\text{RSS}_0$), $F_{\text{jump}} = \frac{(\text{RSS}_0 - \text{RSS}_1)/2}{\text{RSS}_1 / (n_w - n_c)}$ (or Schrader–Hettmansperger robust nested $F$ when `robust != "none"`).
  2. `step_pvalue` (1 df): discontinuous 2-piece linear ($\text{RSS}_1$) vs. optimal continuous hinge across the window ($\text{RSS}_{\text{hinge}} = \min_k \text{RSS}_{\text{hinge}}(k)$), $F_{\text{step}} = \frac{\text{RSS}_{\text{hinge}} - \text{RSS}_1}{\text{RSS}_1 / (n_w - n_c)}$ (or Schrader–Hettmansperger robust nested $F$ when `robust != "none"`).
  **Split-search calibration (`jump_calibration`).** Because $j^*$ maximizes $F_j$ over the $J = n_w - 2\,\text{min\_seg} + 1$ candidate splits in the window, $F(2, n_w - n_c)$ is the wrong reference distribution (the statistic is a sup-$F$; Andrews 1993). With the window's harmonics held fixed, every $F_j$ is a ratio of quadratic forms in the whitened errors, so under Gaussian errors $\max_j F_j$ is pivotal:
  - `"montecarlo"` *(default for non-robust fits)*: draw $B$ (= `n_mc`) standard-normal vectors, project off the null design, compute $F_j$ for all splits at once (2-column Schur complements of the step and slope-change columns), and set $p = (1 + \#\{\max_j F_j^{(b)} \ge F_{\text{obs}}\})/(B + 1)$. When no draw exceeds $F_{\text{obs}}$, the Bonferroni bound (below) capped at $1/(B+1)$ is reported, which leaves the test unchanged for $\alpha \ge 1/(B+1)$.
  - `"bonferroni"` (analytical; default for robust fits): $p = \min\{1,\, J\, p_{\text{raw}}\}$ where $p_{\text{raw}}$ is the fixed-split OLS $F$ or Schrader–Hettmansperger robust $F$ $p$-value.
  - `"none"` (forced by `parity_mode=True`): raw fixed-split screening $p$-values.

  With `"montecarlo"` or `"bonferroni"`, both `jump_pvalue` and `step_pvalue` (Bonferroni-adjusted for the split search) are Holm-adjusted across all candidate locations **before** applying `mag_th`/`dir_th` effect-size filters, so `jump_alpha` (default $0.05$) bounds the probability of reporting any false jump. `step_pvalue` is an optional additional filter (`step_alpha`).
- **Decomposition:** Overlapping windows are combined by Gaussian-weighted averaging with weights $w(t) = \exp\!\left(\frac{\ln 0.1}{(W/6)^2}(t - \tau_0)^2\right)$ (using floating-point sampling rate in default mode so arbitrary time units such as days or seconds never underflow). In non-parity mode, single-point dangling jumps at window edges (`lb + 1` or `ub - 1`) are trimmed from the window's boundaries so every window with a jump has at least two observations per side. Local seasonal components are sample-mean centered only in `parity_mode=True`.

---

## 4. Near-Real-Time Monitoring (`JUSTmonitor`)

*Primary sources: Ghaderpour & Vujadinovic (2020b); Lan & DeMets (1983).*

1. **Historical Model & Causal Preprocessing:** Fit a linear trend plus seasonal harmonics (`ALLSSA` at precision `decimal=2` in default mode or `decimal=1` in `parity_mode`, `OLS`, or `none`) on the stable historical segment $t[\text{history}:\text{start}]$ and evaluate the forecast $\hat{f}(t)$ forward through the monitoring period, yielding forecast residuals $e(t) = f(t) - \hat{f}(t)$. When censored data are present, marginal imputation (`ndimpute`) is fitted strictly causally on $t[\text{history}:\text{start}]$ with monitoring observations imputed sequentially without future look-ahead (emitting `MONITOR_MARGINAL_IMPUTATION` when `imputation_model="trend"` is overridden). Detection steps are strictly causal; optional post-alarm `magnitude` and `direction` estimates use the available active series once an abrupt alarm triggers.
2. **Predictive Residual Covariance (`account_for_forecast_uncertainty=True`):** Under independent Gaussian noise, historical OLS orthogonality implies $\text{Cov}(e_{\text{in}}, e_{\text{out}}) = \mathbf{0}$, while out-of-sample forecast residuals have exact covariance
   $$\Sigma_{\text{out}} = \sigma^2 \Bigl(P_{\text{out}}^{-1} + D_{\text{out}} (D_{\text{his}}^T P_{\text{his}} D_{\text{his}})^{-1} D_{\text{out}}^T\Bigr).$$
   For `season="allssa"`, Gauss–Newton frequency-derivative tangent columns $u (-a_j \sin(2\pi f_j t) + b_j \cos(2\pi f_j t)) / A_j$ are appended to $D$ in the predictive Jacobian to propagate out-of-sample phase-drift uncertainty from data-estimated frequencies. In default mode, `tspec` inverts $\Sigma_{\text{win}} = P_{\text{win}}^{-1} + \operatorname{block\_diag}(0, V_{\text{out}} V_{\text{out}}^T)$ (which reduces to $\text{block\_diag}(P_{\text{in}}, \Sigma_{\text{out}}^{-1})$ for diagonal weights) at each monitoring step (`parity_mode=True` disables this for exact upstream matching).
3. **Sequential Residual LSSA:** At each incoming observation $d = 1, \dots, D$, compute the constant-detrended normalized LSS of $e$ over the rolling window (expanding from $L_h$ or constant-width sliding) across $K = 35$ frequencies $\Omega_{\text{mon}} = \{0.1, 0.2, \dots, 3.5\}$ cy/yr:
   - Low-frequency peak ($\max_{\omega < 1} s(\omega) > s_{\text{crit}}$) $\implies$ **abrupt disturbance** (`type=1`).
   - High-frequency peak exclusively ($\max_{\omega \ge 1} s(\omega) > s_{\text{crit}}$) $\implies$ **phenological disturbance** (`type=2`).
4. **Multiplicity & Sequential-Peeking Control:**
   - **Per-step across-time budget $\alpha_d$** over planned horizon $H$:
     - `pocock`: $\alpha_d = 1 - (1 - \alpha)^{1/H}$ (constant Šidák allocation; conservative because successive statistics are positively correlated).
     - No `horizon` supplied (`pocock` or `obrien_fleming`): open-ended spending $\alpha_d = \alpha / \bigl(C (d+1) \ln^2(d+1)\bigr)$, $C = \sum_{k \ge 2} 1/(k \ln^2 k) \approx 2.1097$, so $\sum_d \alpha_d = \alpha$ for any monitoring length. The horizon is never inferred from the number of observations present, because that would make past thresholds depend on future data.
     - `obrien_fleming`: Lan–DeMets incremental alpha-spending $\alpha_d = \alpha^*(d/H) - \alpha^*((d-1)/H)$ where $\alpha^*(u) = 2\bigl(1 - \Phi(z_{\alpha/2}/\sqrt{u})\bigr)$; steps past $H$ emit `HORIZON_EXCEEDED` and retain the step-$H$ floor.
     - `consecutive_k` with `horizon` $H$: a constant level $\alpha^\ast$, requiring $k$ consecutive exceedances. Under the fitted history model with Gaussian errors (including Gauss–Newton frequency tangents for `season="allssa"`), the monitoring residual process is simulated $B$ = `n_mc` times (history refit in each replicate, future sampling gaps resampled from the history, so no monitoring data is used). Each replicate's step $p$-values $p_{b,d}$ give $a_b = \min_{d \ge k} \max_{d-k < j \le d} p_{b,j}$, and $\alpha^\ast$ lies just below the $\lfloor \alpha (B+1) \rfloor$-th smallest $a_b$, so $P(\text{alarm within } H) \le \alpha$ under the simulated null. Without `horizon`: $\alpha_d = \alpha$, which does **not** bound the familywise rate.
   - **Per-step multi-frequency adjustment** across $K$ scanned frequencies: Šidák $\alpha_{\text{adj}} = 1 - (1 - \alpha_d)^{1/K}$ or Bonferroni $\alpha_{\text{adj}} = \alpha_d / K$, applied to the exact Beta threshold $s_{\text{crit}}(\alpha_{\text{adj}}) = 1 - \alpha_{\text{adj}}^{2/\nu}$.

---

## 5. STPD & Piecewise Linear Trend Estimation (TPTR)

*Primary source: Ghaderpour et al. (2024).*

- **Asymmetric Sequential Two-Step Continuous OLS:** For each interior split $j \in [\text{margin}, n_w - \text{margin})$ ($\text{margin} \ge 2$):
  1. Fit unconstrained 2-parameter OLS on $i \in \{0, \dots, j\}$: $\hat{y}_1(t) = \hat{b}_1 + \hat{a}_1 t$.
  2. Anchor the second segment at $(t_j, \hat{b}_1 + \hat{a}_1 t_j)$ and fit 1-parameter slope $\hat{a}_2$ over $i \in \{j, \dots, n_w - 1\}$.
  3. Repeat in reversed time $t' = t_{n_w - 1} - t$; select the direction with smaller $\|r\|_P$ and record `fb_diff` $= |\tau_f - \tau_b|$.
- **Significance, SNR & NDRI:**
  - **Default exact fixed-knot Student $t$-test ($df = n_w - 3$):** Because the second segment is anchored at $\hat{y}_1(t_j)$, $\hat{a}_1$ and $\hat{a}_2$ are negatively correlated ($\text{Cov}(\hat{a}_1, \hat{a}_2) < 0$). Writing $\hat{a}_1 - \hat{a}_2 = \mathbf{v}^T \mathbf{y}$ in closed $O(n_w)$ form yields the exact variance $\widehat{\text{Var}}(\hat{a}_1 - \hat{a}_2) = \frac{\text{RSS}}{n_w - 3} \sum_{i=1}^{n_w} v_i^2 / w_i$. For `robust != "none"`, each window runs a Huber/biweight sequential scan and evaluates significance via the Schrader–Hettmansperger robust nested $F$-test (`D_null` vs. `D_hinge`).
  - **`parity_mode=True` approximation ($df = n_w - 2$):** Uses Ghaderpour et al. (2024)'s independent-segment approximation $t = |\hat{a}_1 - \hat{a}_2| / \sqrt{\text{SE}_1^2 + \text{SE}_2^2}$.
  - **Search calibration (`tp_calibration`):** the knot $j^*$ and the scan direction are chosen by minimising the RSS, so the $t$ statistic at $j^*$ is a screening statistic. Under the null of a straight line with Gaussian errors (variance $\sigma^2/w_i$) the statistic is pivotal. `"montecarlo"` (default) simulates $B$ = `n_mc` null windows on the observed times and weights and reports $p = (1 + \#\{T^*_b \ge T_{\text{obs}}\})/(B+1)$, falling back to $\min(1,\, 2 J \cdot 2\,\bar F_{t,n_w-3}(T_{\text{obs}}))$ over the $J$ candidate knots and 2 scan directions when no replicate exceeds it. `"bonferroni"` (robust fits) uses $\min(1,\, 2 J\, p_{\text{raw}})$ (accounting for both forward and backward scans). Overlapping window candidates within `tp_th` are clustered and combined via the Simes cluster $p$-value $\min_{1 \le j \le c}(c / j)\, p_{(j)}$ before Holm step-down across merged clusters.
  - Weighted $\text{SNR}_{\text{raw}} = \|\hat{y}\|_P / \|f - \hat{y}\|_P$ and shift-invariant $\text{SNR}_{\text{centered}} = \|\hat{y} - \bar{\hat{y}}_P\|_P / \|f - \hat{y}\|_P$.
  - Weighted $\text{NDRI}_{L_2} = \frac{|\|r_1\|_P - \|r_2\|_P|}{\|r_1\|_P + \|r_2\|_P}$ and whitened density-invariant $\text{NDRI}_{\text{RMSE}} = \frac{|\text{RMSE}_1 - \text{RMSE}_2|}{\text{RMSE}_1 + \text{RMSE}_2}$ where $\text{RMSE}_k = \|r_k\|_P / \sqrt{n_k}$.

---

## 6. Censored-Data Imputation (Tobit MLE & `ndimpute`)

*Primary sources: Tobin (1958); Helsel (2012); Turnbull (1976); Fullard (2026).*

- **Default Trend-Aware Censored Regression (`imputation_model="trend"`):** `tspec` fits a scale-invariant Tobit-type maximum-likelihood regression (`_cenreg.py`) conditional on time (linear trend plus optional `trend_harmonics`) directly on the observed, left-censored, right-censored, and interval-censored likelihood:
  $$L(\beta, \sigma) = \prod_{\text{obs}} \frac{1}{\sigma}\phi\!\left(\frac{y_i - \mu_i}{\sigma}\right) \prod_{\text{left}} \Phi\!\left(\frac{L_i - \mu_i}{\sigma}\right) \prod_{\text{right}} \bar\Phi\!\left(\frac{R_i - \mu_i}{\sigma}\right) \prod_{\text{int}} \Bigl[\Phi\!\left(\frac{u_i - \mu_i}{\sigma}\right) - \Phi\!\left(\frac{l_i - \mu_i}{\sigma}\right)\Bigr]$$
  on $y$ (`dist="normal"`) or $\log y$ (`dist="lognormal"`, with `dist="auto"` comparing exact Jacobian-adjusted observed-scale negative log-likelihoods $\text{NLL}_z + \sum_{\text{obs}} \log y_i$ on positive data). Deep tails ($b < -30$) use log-space survival ratios and Gordon's (1941) asymptotic inverse Mills expansion; stochastic draws sample from the exact truncated distribution.
- **Marginal Imputation via `ndimpute` (`imputation_model="marginal"`, or `scope="windowed"` / `"seasonal_stratified"` / `dist="weibull"`):**
  - *Left-censored ROS:* Hirsch–Stedinger / Helsel–Cohn plotting positions with OLS probability-plot regression (`NADA::ros`-exact). `tspec` permutes tied detection-limit groups in `'quantile'` mode and defaults to `'stochastic'` to eliminate chronological rank-order trend injection.
  - *Parametric MLE:* Full censored likelihood (`survival::survreg`-exact) with numerically stable truncated expectations and draws.

---

## 7. Robust M/MM-Estimation, Continuous $L_1$ Splines & Theil–Sen / ATS

*Primary sources: Huber (1981); Schrader & Hettmansperger (1980); Yohai (1987); Sen (1968); Akritas, Murphy, & LaValley (1995); Gilbert (1987).*

- **Huber IRLS (`robust="huber"`, $k = 1.345$):** Minimizes $\sum_{i=1}^n \rho_k(\tilde{r}_i / \hat{s})$ in base-whitened coordinates $\tilde{r} = W r$ (applying IRLS weights directly to the whitened rows $W_{\text{eff}} = \operatorname{diag}(\sqrt{w_{\text{IRLS}}}) W$, exact for both 1D and 2D SPD weights $P = W^T W$), where $\rho_k(u) = \tfrac{1}{2}u^2$ for $|u| \le k$ and $k|u| - \tfrac{1}{2}k^2$ otherwise, with finite-sample-adjusted MAD scale $\hat{s} = \sqrt{\frac{n}{n - p}} \cdot 1.4826\,\text{median}(|\tilde{r}_i|)$ and iterative weights $w_i = \min(1, k\hat{s}/|\tilde{r}_i|)$. Asymptotic parameter covariance follows Huber (1981, Eq. 7.6.11):
  $$\widehat{\text{Cov}}(\hat{c}) = \hat{\tau}^2 (D^T P D)^+, \qquad \hat{\tau}^2 = K_{\text{corr}}^2 \frac{\hat{s}^2 \sum \psi(\tilde{r}_i/\hat{s})^2}{(n - p)\, m^2}, \qquad m = \frac{1}{n}\sum \psi'(\tilde{r}_i/\hat{s}), \qquad K_{\text{corr}} = 1 + \frac{p}{n}\frac{\operatorname{Var}(\psi')}{m^2}.$$
  For Huber's $\psi$, $\psi' \in \{0, 1\}$ so $\operatorname{Var}(\psi')/m^2 = (1 - m)/m$.
- **MM-type Biweight (`robust="mm"`, $c = 4.685061$):** Unlike Yohai's (1987) MM-estimator (which starts from a high-breakdown S-estimate), initializes from convex Huber IRLS (preserving sustained step jumps without masking), so its breakdown point is design-dependent (~35% sustained contamination at one end of a record can break it). It then refines at the fixed Huber scale $\hat{s}$ using Tukey's bounded bisquare loss $\rho_c(u) = \frac{c^2}{6}\bigl(1 - (1 - \min(1, |u|/c)^2)^3\bigr)$ and redescending weights $w_i = \bigl(1 - (u_i/c)^2\bigr)_+^2$, using the general $\operatorname{Var}(\psi')/m^2$ factor in $K_{\text{corr}}$ and giving 95% Gaussian efficiency with **zero influence** for extreme isolated spikes ($|u_i| \ge c$). In STPD window scans, `robust="mm"` uses the convex Huber sequential scan for knot localization and the Tukey biweight Schrader–Hettmansperger test for significance.
- **Schrader–Hettmansperger (1980) Robust Nested $F$-Test:** For nested designs $D_{\text{null}} \subset D_{\text{full}}$ evaluated at the **shared full-model scale** $\hat{s}_1$:
  $$F_{\text{rob}} = \frac{2\bigl(L_{\text{null}}(\hat{s}_1) - L_{\text{full}}(\hat{s}_1)\bigr) / q}{\hat{\lambda}}, \qquad \hat{\lambda} = K_{\text{corr}} \frac{\sum \psi(\tilde{r}_{1,i}/\hat{s}_1)^2}{(n - p_1)\, m_1}, \qquad p = 1 - F_{q,\, n - p_1}(F_{\text{rob}}).$$
- **Exact Continuous $L_1$ (LAD) Hinge-Spline (`mode="l1"`):** Minimizes $\sum_{i=1}^n w_i \bigl|y_i - (b_0 + a_0 u_i + \sum_{k=1}^m \delta_k (u_i - u_{\tau_k})_+)\bigr|$ globally via sparse HiGHS linear programming (`scipy.optimize.linprog`), guaranteeing strict continuity at every turning point (emitting `L1_SOLVER_FALLBACK` if HiGHS falls back to IRLS). Robust to isolated vertical outliers; its breakdown point is design-dependent (about 0.29 for an intercept + slope on a uniform design, He et al. 1990), not 50%.
- **Theil–Sen & Akritas–Theil–Sen (`slope_estimator="sen"`):**
  - *Uncensored segments:* Pairwise median slope $\hat{\beta} = \text{median}_{i < j}\frac{y_j - y_i}{t_j - t_i}$, median intercept $\hat{\alpha} = \text{median}_i(y_i - \hat{\beta} t_i)$, tie-adjusted Gilbert (1987, Eq. 16.8) order-statistic 95% confidence interval, and two-sided Mann–Kendall $p$-value.
  - *Left/right-censored segments (`<DL` / `>UL`):* Automatically computes the **Akritas–Theil–Sen (ATS)** slope solving $S_{\text{cen}}(\beta) = 0$ on censored residual concordances $r_i(\beta) = y_i - \beta t_i$, with score-inverted confidence bounds $S_{\text{cen}}(\beta_{\text{lo}}) = +z_{1-\alpha/2}\sqrt{\text{Var}(S)}$ and $S_{\text{cen}}(\beta_{\text{hi}}) = -z_{1-\alpha/2}\sqrt{\text{Var}(S)}$. $\text{Var}(S)$ is the censoring-aware permutation variance of Daniels' (1944) generalized correlation: with $a_{ij} = \text{sign}(t_j - t_i)$ and $b_{ij} \in \{-1, 0, 1\}$ the definite concordance of the residual responses, $\text{Var}(S) = \tfrac14\bigl[\tfrac{2 A_2 B_2}{n(n-1)} + \tfrac{4 (A_3 - A_2)(B_3 - B_2)}{n(n-1)(n-2)}\bigr]$ with $A_2 = \sum a_{ij}^2$, $A_3 = \sum_i (\sum_j a_{ij})^2$ (likewise $B$). It equals $n(n-1)(2n+5)/18$ without censoring and is smaller with censoring; it is evaluated at slope 0 for the $p$-value and at $\hat\beta$ for the CI. When `mk_correction` is `"ar1"` or `"hamed_rao"`, the autocorrelation variance inflation (and AR(1) bootstrap for short/autocorrelated series) is applied to censored ATS segments as well.

---

## 8. Analytical & Bootstrap Confidence and Prediction Intervals

All analytical pointwise intervals at nominal level $1 - \alpha = \text{ci\_level}$ assume independent residuals unless a full inverse-covariance weight matrix $P = C_f^{-1}$ is supplied, and are conditional on the detected breakpoints and selected harmonic frequencies. The default post-detection bands of `stpd` and `analyze_series` (`band_method="bootstrap"`, item 7) propagate turning-point, step-offset, and seasonal estimation uncertainty:

1. **JUST Decomposition (`just_decompose`):** Within each local window $m$ with unscaled parameter covariance $\hat{\Sigma}_m$ (from WLS or Huber/MM IRLS), default mode (`parity_mode=False`) keeps the fitted pure sinusoids $\hat{S}_m(t) = D_{\text{seas}} \hat{c}_S$ unshifted, giving exact block variances:
   $$\widehat{\text{Var}}(\hat{T}_m(t_i)) = \bigl(D_{\text{tr}} \hat{\Sigma}_{m,TT} D_{\text{tr}}^T\bigr)_{ii}, \qquad \widehat{\text{Var}}(\hat{S}_m(t_i)) = \bigl(D_{\text{seas}} \hat{\Sigma}_{m,SS} D_{\text{seas}}^T\bigr)_{ii}.$$
   In `parity_mode=True` (within-window sample-mean centering), the variance uses the centered seasonal basis $D_{\text{seas}}^{\text{cent}} = D_{\text{seas}} - \mathbf{1}\bar{\mathbf{d}}_{\text{seas}}^T$ and shifted trend basis $D_{\text{tr,eff}} = [D_{\text{tr}},\, \mathbf{1}\bar{\mathbf{d}}_{\text{seas}}^T]$. Per-window half-widths $\text{hw}_m(t_i) = t_{1-\alpha/2,\, n_w - n_c}\sqrt{\widehat{\text{Var}}_m(t_i)}$ are combined using the conservative **Cauchy–Schwarz upper-bound blend**:
   $$\text{hw}(t_i) = \frac{\sum_m g_m(t_i)\,\text{hw}_m(t_i)}{\sum_m g_m(t_i)}.$$

   **Bias-corrected trend & seasonal bands (`trend_ci_method="bias_corrected"`, default).** For the trend band, each window is refit with an added quadratic trend term (piecewise intercepts/slopes at local jumps plus global $t^2$), centering the band on `trend_bc` (Calonico, Cattaneo & Farrell 2018). For `season="allssa"`, the seasonal band refits each window on the union of ALLSSA-selected frequencies and candidate intra-annual harmonics ($\ge 1$ cy/yr, subject to available DOF) so weak unselected harmonics do not cause post-selection under-coverage, centering the band on `seasonal_bc`. `plot_just_decompose(..., show_ci=True)` plots `trend_bc` and `seasonal_bc` alongside their bands.
2. **Connected Piecewise Linear Trends (`fit_piecewise_trend` & `detect_turning_points`):**
   - *`mode="joint"` / `"huber"`:* Exact $\widehat{\text{Var}}(\hat{y}_i) = \mathbf{d}_i^T \widehat{\text{Cov}}(\hat{c})\,\mathbf{d}_i$ with $t_{1-\alpha/2,\, n - \text{rank}}$.
   - *`mode="sequential"` (TPTR):* For segment $0$ ($i \le \tau_1$), $V_i = S_{w,0}^{-1} + (t_i - \bar{t}_0)^2 / S_{tt,0}$. For each anchored segment $k \ge 1$ ($i \in [\tau_k, \tau_{k+1}]$) with $d_i = t_i - t_{\tau_k}$, $S_{dd,k} = \sum w_j d_j^2$, and $\gamma_k = (\sum w_j d_j) / S_{dd,k}$,
     $$V_i = (1 - d_i \gamma_k)^2 V_{\tau_k} + \frac{d_i^2}{S_{dd,k}}, \qquad \text{SE}(\hat{y}_i) = \sqrt{\frac{\text{RSS}_W}{n - (m + 2)} V_i}.$$
   - *`mode="l1"`:* Analytical `fit_piecewise_trend(..., mode="l1")` returns `None` (no closed-form finite-sample SE); `detect_turning_points(..., refit_mode="l1", band_method="bootstrap")` produces wild-bootstrap bands with exact-zero LP vertices unpinned.
3. **Near-Real-Time Monitoring (`just_monitor`):** From the historical fit covariance $\hat{\Sigma}_{\text{his}}$ and base observation residual variance $\hat{\sigma}_{\text{his}}^2$ (weighted RSS / $\text{dof}_{\text{his}}$ for WLS; df-adjusted $\text{MAD}^2$ for `robust != "none"`), the mean forecast CI and out-of-sample observation prediction interval are:
   $$\text{CI}(t_i) = \hat{f}(t_i) \pm t_{1-\alpha/2,\, \text{dof}_{\text{his}}} \sqrt{\mathbf{d}_i^T \hat{\Sigma}_{\text{his}} \mathbf{d}_i}, \qquad \text{PI}(t_i) = \hat{f}(t_i) \pm t_{1-\alpha/2,\, \text{dof}_{\text{his}}} \sqrt{\mathbf{d}_i^T \hat{\Sigma}_{\text{his}} \mathbf{d}_i + \hat{\sigma}_{\text{his}}^2 (P^{-1})_{ii}}.$$
4. **Theil–Sen / Akritas–Theil–Sen slope fan (`analyze_series(..., slope_estimator="sen")`):** exported as `slope_fan_low/high` (not `trend_ci_*`), because it has zero width at each segment's median time and is **not** a pointwise CI for the trend. Each segment plots the median-pivoted slope envelope around $(t_{\text{med}},\, \hat{\alpha} + \hat{\beta}\, t_{\text{med}})$ bounded by $\hat{y}_{\text{med}} + \hat{\beta}_{\text{lo}}(t_i - t_{\text{med}})$ and $\hat{y}_{\text{med}} + \hat{\beta}_{\text{hi}}(t_i - t_{\text{med}})$ (Helsel, 2012).
5. **Censored segments (`analyze_series`, `slope_estimator="ols"`):** For segments containing censored values, `n_imputations` (default 20) proper multiple imputations are drawn from a censored linear regression in time fitted to the deseasonalized values and limits using the resolved distribution (`"lognormal"` via exact Jacobian-adjusted observed-scale NLL when requested/selected on strictly positive data, else `"normal"`, with parameters drawn from their asymptotic normal distribution in correlation space). Each imputed segment is fit by OLS (with HC3 heteroscedasticity-robust variance when `dist="lognormal"`), and slopes are pooled with Rubin's rules, $T = \bar{U} + (1 + 1/m) B$, with Barnard–Rubin (1999) degrees of freedom (`estimator="ols_mi"`).
6. **Censored detection p-values (`detection_imputations` $= M > 1$):** each candidate window is imputed $M$ times from a censored-regression fit of the window's null model with parameters drawn from their asymptotic normal distribution. Each imputation yields a calibrated $p$-value $p_m$ (at its own best split/knot for non-robust fits, or at the primary candidate split for robust `just_jumps`), and $z_m = \Phi^{-1}(1 - p_m)$ with $p_m$ clipped to $[10^{-15},\, 1 - 10^{-12}]$. The pooled $p$-value is $1 - F_{t,\nu}\bigl(\bar z / \sqrt{1 + r}\bigr)$ with $r = (1 + 1/M) B$, $B$ the between-imputation variance of $z_m$, and $\nu = (M-1)(1 + 1/r)^2$ (Licht, 2010; van Buuren, 2018, §5.3). For Bonferroni calibrations the fixed-split $p$-values are pooled first and then multiplied by the number of candidate splits. Pooled $p$-values are Holm-adjusted across candidates as usual.
7. **Detect-then-fit bootstrap bands (`band_method="bootstrap"`):** with fitted trend $\hat f$, other fitted structure $\hat g$ (seasonal, jump steps) and raw residuals $e = y - \hat f - \hat g$, leverage-adjusted wild-bootstrap residuals are $e_i^* = e_i / \sqrt{1 - h_{ii}}$ using exact hat diagonals of the hinge trend design and windowed seasonal estimator (falling back to $e_i \sqrt{n/(n - p_{\text{eff}})}$ if $\max h_{ii} \ge 0.99$), and replicates are $y^*_b = \hat f + \hat g + e^* \odot v_b$ with Rademacher signs $v_b \in \{-1, +1\}$ (for `refit_mode="l1"`, exact-zero LP interpolation points draw replacement magnitudes from the non-zero residuals). In each replicate:
   - `analyze_series` re-runs its windowed seasonal estimator (with robust observation weights when `robust != "none"`, preventing isolated outliers from inflating replicate variance);
   - under `jump_adjustment="offset"`, pre-STPD local-window step offsets are removed, turning points are re-located with STPD's knot estimator over their detecting windows, and a unified joint step+hinge refit estimates the final curve; under `jump_adjustment="segment"`, each inter-jump epoch is re-located and refitted independently;
   - OLS segment slopes are re-estimated between the replicate's breakpoints (censored `"ols_mi"` segments retain Rubin-pooled MI CIs).

   Where fixed-structure bootstrap dispersion falls below the analytical conditional SE, deviations $d = \theta^*_b - \theta_{\text{boot}}$ are scaled pointwise by $\max\{1, \min(2, \mathrm{SE}_{\text{analytic}} / \mathrm{SD}_{\text{fixed}})\}$. Trend bands are percentile intervals $[\hat\theta + q_{\alpha/2}(d), \hat\theta + q_{1-\alpha/2}(d)]$; segment-slope CIs are basic intervals $[\hat s - q_{1-\alpha/2}(d), \hat s - q_{\alpha/2}(d)]$. All are conditional on the number of breakpoints detected.

---

## 9. Serial Correlation in Mann–Kendall / Theil–Sen

`analyze_series(..., slope_estimator="sen", mk_correction="ar1")` (default) estimates the lag-1 autocorrelation on robustly detrended residuals via the Gaussian-mapped Spearman rank lag-1 correlation $\hat\rho_S = 2\sin(\pi r_S / 6)$ (resistant to isolated outliers, with shift/scale-invariant rank seeding) and inflates $\text{Var}(S)$ by $n/n^* = 1 + \frac{2}{n(n-1)(n-2)}\sum_{k=1}^{n-1}(n-k)(n-k-1)(n-k-2)\,\rho^S_k$, where $\rho^S_k = \tfrac{6}{\pi}\arcsin(\hat\rho^k/2)$.

When $n \le 1000$ and the autocorrelation-adjusted effective sample size $n_{\text{eff}} < 150$ (or $n \le 300$), `tspec` debiases $\hat\rho_S$ by common-random-numbers bisection so the simulated expectation matches the observed rank lag-1 correlation (with a smooth small-$n$ guard for $n < 50$), draws AR(1) bootstrap series around $\hat\rho$ (with estimation uncertainty $\sqrt{(1-\rho^2)/n}$), and studentizes each draw with its own re-estimated variance factor; the observed $|z|$ is referred to the simulated studentized null distribution for the $p$-value (using $\min(p_{\text{MC}}, p_{\text{normal}})$ below Monte Carlo resolution), and its $1 - \alpha$ quantile replaces $z_{1-\alpha/2}$ in the Sen/ATS confidence interval. Larger effective samples use the analytical inflation factor directly. `mk_correction="hamed_rao"` uses the significant-lags estimate of Hamed & Rao (1998), which matches `pymannkendall` closely but stays liberal in small samples; `"none"` assumes independence.
