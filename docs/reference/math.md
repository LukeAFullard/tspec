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
4. Accept $\omega_{\text{new}}$ only if it is significant **after accounting for the frequency search**; terminate when $F$ stabilizes (with explicit `max_iter` and cycle guards). `significance` selects the rule:
   - `"montecarlo"` *(default)*: under Gaussian white noise the normalized LSS is invariant to the error scale and the known-constituent coefficients, so drawing $B$ (= `n_mc`) standard-normal vectors in whitened coordinates, projecting off $\text{col}(WD)$ and taking $\max_{\omega \in G} s(\omega)$ over a dense grid $G$ covering every frequency the search can visit gives the exact null distribution of the searched maximum. Accept if $p_{\text{FW}} = (1 + \#\{\max^{(b)} \ge s(\omega_{\text{new}})\})/(B + 1) \le \alpha$.
   - `"sidak"`: accept if $s(\omega_{\text{new}}) > s_{\text{crit}}(\alpha_K)$ with $\alpha_K = 1 - (1 - \alpha)^{1/K_{\text{eff}}}$ and $K_{\text{eff}} \approx T \times$ (searched bandwidth) independent frequencies (Rayleigh resolution $1/T$), capped by the grid size.
   - `"per_frequency"` (published rule; forced by `parity_mode=True`): $s(\omega_{\text{new}}) > s_{\text{crit}}(\alpha)$. This ignores the search and selects spurious cycles in roughly 20% of pure-noise series at $\alpha = 0.01$.

`lssa()` reports `peak_fap`, the family-wise false-alarm probability of the highest peak over the supplied grid (`"sidak"` by default, or `"montecarlo"`); per-frequency `pvalues` are unchanged.

---

## 3. JUST Jump Detection & Decomposition

*Primary sources: Ghaderpour & Vujadinovic (2020a); Ghaderpour (2021).*

- **Windowed 2-Piece Scan:** Within each sliding window (count-based `size=3M, step=M` or calendar `window_duration`), scan $j \in \{3, \dots, n_w - 3\}$, fitting a discontinuous 2-piece linear trend jointly with ALLSSA/OLS harmonics, selecting $j^*$ minimizing $\|r\|_P^2$:
  $$\text{Mag} = (\hat{b}_2 - \hat{b}_1) + (\hat{a}_2 - \hat{a}_1) t_{0,j^*}, \qquad \text{Dir} = \hat{a}_2 - \hat{a}_1.$$
- **Nested Partial $F$-Tests (clean-room additions):** Selecting harmonics under the **no-jump null** model first (for `season="allssa"`, preventing harmonic-mining inflation under $H_0$) and holding those harmonics fixed across nested models $D_{\text{null}} \subset D_{\text{hinge}} \subset D_{\text{full}}$, `tspec` computes:
  1. `jump_pvalue` (2 df): discontinuous 2-piece linear ($\text{RSS}_1$) vs. continuous 1-piece linear ($\text{RSS}_0$), $F_{\text{jump}} = \frac{(\text{RSS}_0 - \text{RSS}_1)/2}{\text{RSS}_1 / (n_w - n_c)}$.
  2. `step_pvalue` (1 df): discontinuous 2-piece linear ($\text{RSS}_1$) vs. optimal continuous hinge across the window ($\text{RSS}_{\text{hinge}} = \min_k \text{RSS}_{\text{hinge}}(k)$), $F_{\text{step}} = \frac{\text{RSS}_{\text{hinge}} - \text{RSS}_1}{\text{RSS}_1 / (n_w - n_c)}$.
  **Split-search calibration (`jump_calibration`).** Because $j^*$ maximizes $F_j$ over the window, $F(2, n_w - n_c)$ is the wrong reference distribution (the statistic is a sup-$F$; Andrews 1993). With the window's harmonics held fixed, every $F_j$ is a ratio of quadratic forms in the whitened errors, so under Gaussian errors $\max_j F_j$ is pivotal:
  - `"montecarlo"` *(default for non-robust fits)*: draw $B$ (= `n_mc`) standard-normal vectors, project off the null design, compute $F_j$ for all splits at once (2-column Schur complements of the step and slope-change columns), and set $p = (1 + \#\{\max_j F_j^{(b)} \ge F_{\text{obs}}\})/(B + 1)$. When no draw exceeds $F_{\text{obs}}$, the Bonferroni bound (below) capped at $1/(B+1)$ is reported, which leaves the test unchanged for $\alpha \ge 1/(B+1)$.
  - `"bonferroni"` (analytical; default for robust fits): $p = \min\{1, (n_w - 5)\, p_{F(2,\,n_w - n_c)}\}$.
  - `"none"` (forced by `parity_mode=True`): raw fixed-split screening $p$-values.

  With `"montecarlo"` or `"bonferroni"`, `jump_pvalue` is then Holm-adjusted across the distinct candidate locations, so `jump_alpha` (default $0.05$) bounds the probability of reporting any false jump. `step_pvalue` is an optional additional filter (`step_alpha`).
- **Decomposition:** Overlapping windows are combined by Gaussian-weighted averaging with weights $w(t) = \exp\!\left(\frac{\ln 0.1}{(W/6)^2}(t - \tau_0)^2\right)$ (using floating-point sampling rate in default mode so arbitrary time units such as days or seconds never underflow). Local seasonal components are sample-mean centered only in `parity_mode=True`.

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
     - `pocock`: $\alpha_d = 1 - (1 - \alpha)^{1/H}$ (constant Šidák allocation; not Pocock's group-sequential boundary, and conservative because successive statistics are positively correlated).
     - No `horizon` supplied (`pocock` or `obrien_fleming`): open-ended spending $\alpha_d = \alpha / \bigl(C (d+1) \ln^2(d+1)\bigr)$, $C = \sum_{k \ge 2} 1/(k \ln^2 k) \approx 2.1097$, so $\sum_d \alpha_d = \alpha$ for any monitoring length. The horizon is never inferred from the number of observations present, because that would make past thresholds depend on future data.
     - `obrien_fleming`: Lan–DeMets incremental alpha-spending $\alpha_d = \alpha^*(d/H) - \alpha^*((d-1)/H)$ where $\alpha^*(u) = 2\bigl(1 - \Phi(z_{\alpha/2}/\sqrt{u})\bigr)$; steps past $H$ emit `HORIZON_EXCEEDED` and retain the step-$H$ floor.
     - `consecutive_k`: $\alpha_d = \alpha$, requiring $k$ consecutive exceedances. This does **not** bound the familywise rate (about 10% over 96 steps at $\alpha = 0.05$ in simulation).
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

---

## 7. Robust M/MM-Estimation, Continuous $L_1$ Splines & Theil–Sen / ATS

*Primary sources: Huber (1981); Schrader & Hettmansperger (1980); Yohai (1987); Sen (1968); Akritas, Murphy, & LaValley (1995); Gilbert (1987).*

- **Huber IRLS (`robust="huber"`, $k = 1.345$):** Minimizes $\sum_{i=1}^n \rho_k(\tilde{r}_i / \hat{s})$ in base-whitened, median-centered coordinates where $\rho_k(u) = \tfrac{1}{2}u^2$ for $|u| \le k$ and $k|u| - \tfrac{1}{2}k^2$ otherwise, with finite-sample-adjusted MAD scale $\hat{s} = \sqrt{\frac{n}{n - p}} \cdot 1.4826\,\text{median}(|\tilde{r}_i|)$ and iterative weights $w_i = \min(1, k\hat{s}/|\tilde{r}_i|)$. Asymptotic parameter covariance follows Huber (1981, Eq. 7.6.11):
  $$\widehat{\text{Cov}}(\hat{c}) = \hat{\tau}^2 (D^T P D)^+, \qquad \hat{\tau}^2 = \left(1 + \frac{p}{n}\frac{1 - m}{m}\right)^2 \frac{\hat{s}^2 \sum \psi(\tilde{r}_i/\hat{s})^2}{(n - p)\, m^2}, \qquad m = \frac{1}{n}\sum \psi'(\tilde{r}_i/\hat{s}).$$
- **MM-type Biweight (`robust="mm"`, $c = 4.685$):** Unlike Yohai's (1987) MM-estimator (which starts from a high-breakdown S-estimate), initializes from convex Huber IRLS, so its breakdown point is design-dependent (it fails under ~35% sustained vertical contamination at one end of a record). It (preserving sustained step jumps without masking) and refines at the fixed scale $\hat{s}$ using Tukey's bounded bisquare loss $\rho_c(u) = \frac{c^2}{6}\bigl(1 - (1 - \min(1, |u|/c)^2)^3\bigr)$ and redescending weights $w_i = \bigl(1 - (u_i/c)^2\bigr)_+^2$, giving 95% Gaussian efficiency with **zero influence** for extreme isolated spikes ($|u_i| \ge c$).
- **Schrader–Hettmansperger (1980) Robust Nested $F$-Test:** For nested designs $D_{\text{null}} \subset D_{\text{full}}$ evaluated at the **shared full-model scale** $\hat{s}_1$:
  $$F_{\text{rob}} = \frac{2\bigl(L_{\text{null}}(\hat{s}_1) - L_{\text{full}}(\hat{s}_1)\bigr) / q}{\hat{\lambda}}, \qquad \hat{\lambda} = K_{\text{corr}} \frac{\sum \psi(\tilde{r}_{1,i}/\hat{s}_1)^2}{(n - p_1)\, m_1}, \qquad p = 1 - F_{q,\, n - p_1}(F_{\text{rob}}).$$
- **Exact Continuous $L_1$ (LAD) Hinge-Spline (`mode="l1"`):** Minimizes $\sum_{i=1}^n w_i \bigl|y_i - (b_0 + a_0 u_i + \sum_{k=1}^m \delta_k (u_i - u_{\tau_k})_+)\bigr|$ globally via sparse HiGHS linear programming (`scipy.optimize.linprog`), guaranteeing strict continuity at every turning point. Robust to isolated vertical outliers; its breakdown point is
    design-dependent (about 0.29 for an intercept + slope on a uniform design, He et al. 1990),
    not 50%.
- **Theil–Sen & Akritas–Theil–Sen (`slope_estimator="sen"`):**
  - *Uncensored segments:* Pairwise median slope $\hat{\beta} = \text{median}_{i < j}\frac{y_j - y_i}{t_j - t_i}$, median intercept $\hat{\alpha} = \text{median}_i(y_i - \hat{\beta} t_i)$, tie-adjusted Gilbert (1987, Eq. 16.8) order-statistic 95% confidence interval, and two-sided Mann–Kendall $p$-value.
  - *Left/right-censored segments (`<DL` / `>UL`):* Automatically computes the **Akritas–Theil–Sen (ATS)** slope solving $S_{\text{cen}}(\beta) = 0$ on censored residual concordances $r_i(\beta) = y_i - \beta t_i$, with score-inverted confidence bounds $S_{\text{cen}}(\beta_{\text{lo}}) = +z_{1-\alpha/2}\sqrt{\text{Var}(S)}$ and $S_{\text{cen}}(\beta_{\text{hi}}) = -z_{1-\alpha/2}\sqrt{\text{Var}(S)}$. $\text{Var}(S)$ is the censoring-aware permutation variance of Daniels' (1944) generalized correlation: with $a_{ij} = \text{sign}(t_j - t_i)$ and $b_{ij} \in \{-1, 0, 1\}$ the definite concordance of the residual responses, $\text{Var}(S) = \tfrac14\bigl[\tfrac{2 A_2 B_2}{n(n-1)} + \tfrac{4 (A_3 - A_2)(B_3 - B_2)}{n(n-1)(n-2)}\bigr]$ with $A_2 = \sum a_{ij}^2$, $A_3 = \sum_i (\sum_j a_{ij})^2$ (likewise $B$). It equals $n(n-1)(2n+5)/18$ without censoring and is smaller with censoring; it is evaluated at slope 0 for the $p$-value and at $\hat\beta$ for the CI.

---

## 8. Analytical Pointwise Confidence & Prediction Intervals

All pointwise intervals at nominal level $1 - \alpha = \text{ci\_level}$ are conditional on the detected breakpoints and selected harmonic frequencies and assume independent residuals unless a full inverse-covariance weight matrix $P = C_f^{-1}$ is supplied:

1. **JUST Decomposition (`just_decompose`):** Within each local window $m$ with unscaled parameter covariance $\hat{\Sigma}_m$ (from WLS or Huber/MM IRLS), default mode (`parity_mode=False`) keeps the fitted pure sinusoids $\hat{S}_m(t) = D_{\text{seas}} \hat{c}_S$ unshifted (preventing seasonal-balance bias on irregular timestamps where windows contain unequal summer/winter counts), giving exact block variances:
   $$\widehat{\text{Var}}(\hat{T}_m(t_i)) = \bigl(D_{\text{tr}} \hat{\Sigma}_{m,TT} D_{\text{tr}}^T\bigr)_{ii}, \qquad \widehat{\text{Var}}(\hat{S}_m(t_i)) = \bigl(D_{\text{seas}} \hat{\Sigma}_{m,SS} D_{\text{seas}}^T\bigr)_{ii}.$$
   In `parity_mode=True` (Ghaderpour's within-window sample-mean centering $\hat{T}_m \leftarrow \hat{T}_m + \bar{s}_m$, $\hat{S}_m \leftarrow \hat{S}_m - \bar{s}_m$), the variance uses the centered seasonal basis $D_{\text{seas}}^{\text{cent}} = D_{\text{seas}} - \mathbf{1}\bar{\mathbf{d}}_{\text{seas}}^T$ and shifted trend basis $D_{\text{tr,eff}} = [D_{\text{tr}},\, \mathbf{1}\bar{\mathbf{d}}_{\text{seas}}^T]$.
   Because adjacent Gaussian windows share overlapping observations (positive correlation), per-window half-widths $\text{hw}_m(t_i) = t_{1-\alpha/2,\, n_w - n_c}\sqrt{\widehat{\text{Var}}_m(t_i)}$ are combined using the conservative **Cauchy–Schwarz upper-bound blend**:
   $$\text{hw}(t_i) = \frac{\sum_m g_m(t_i)\,\text{hw}_m(t_i)}{\sum_m g_m(t_i)},$$
   which upper-bounds the standard deviation of the blended estimator (since $\text{SD}(\sum a_m X_m) \le \sum a_m \text{SD}(X_m)$). This bound controls variance only: a local-linear window fit is biased by $\approx \tfrac12 h^2 \mu_2 T''(t)$ when the trend curves within a window, and that bias does not shrink with $n$.

   **Bias-corrected trend band (`trend_ci_method="bias_corrected"`, default).** Each window is refit with an added quadratic trend term (piecewise intercepts and slopes at the local jumps plus one global $t^2$), holding the selected harmonics fixed. The trend band is centered on the blended local-quadratic estimate (`trend_bc`) with the blended half-widths of that richer model, i.e. robust bias correction (Calonico, Cattaneo & Farrell 2018). In simulations its pointwise coverage stays at about 95% for linear, jump and smoothly curved trends (e.g. a 9-year cycle), whereas the `"model"` band (forced by `parity_mode=True`) falls to about 70%. The reported point estimate `trend` is unchanged, so the band need not be centered on it.
2. **Connected Piecewise Linear Trends (`fit_piecewise_trend` & `detect_turning_points`):**
   - *`mode="joint"` / `"huber"`:* Exact $\widehat{\text{Var}}(\hat{y}_i) = \mathbf{d}_i^T \widehat{\text{Cov}}(\hat{c})\,\mathbf{d}_i$ with $t_{1-\alpha/2,\, n - \text{rank}}$.
   - *`mode="sequential"` (TPTR):* For segment $0$ ($i \le \tau_1$), $V_i = S_{w,0}^{-1} + (t_i - \bar{t}_0)^2 / S_{tt,0}$. For each anchored segment $k \ge 1$ ($i \in [\tau_k, \tau_{k+1}]$) with $d_i = t_i - t_{\tau_k}$, $S_{dd,k} = \sum w_j d_j^2$, and $\gamma_k = (\sum w_j d_j) / S_{dd,k}$, the slope estimator is supported strictly on $\{\tau_k + 1, \dots, \tau_{k+1}\}$ (disjoint from the anchor $\hat{y}_{\tau_k}$), giving the exact $O(n)$ recursion:
     $$V_i = (1 - d_i \gamma_k)^2 V_{\tau_k} + \frac{d_i^2}{S_{dd,k}}, \qquad \text{SE}(\hat{y}_i) = \sqrt{\frac{\text{RSS}_W}{n - (m + 2)} V_i}.$$
   - *`mode="l1"`:* Returns `None` (no closed-form finite-sample SE for LP $L_1$ regression).
3. **Near-Real-Time Monitoring (`just_monitor`):** From the historical fit covariance $\hat{\Sigma}_{\text{his}}$ and base observation residual variance $\hat{\sigma}_{\text{his}}^2$ (weighted RSS / $\text{dof}_{\text{his}}$ for WLS; df-adjusted $\text{MAD}^2$ for `robust != "none"`), the mean forecast CI and out-of-sample observation prediction interval are:
   $$\text{CI}(t_i) = \hat{f}(t_i) \pm t_{1-\alpha/2,\, \text{dof}_{\text{his}}} \sqrt{\mathbf{d}_i^T \hat{\Sigma}_{\text{his}} \mathbf{d}_i}, \qquad \text{PI}(t_i) = \hat{f}(t_i) \pm t_{1-\alpha/2,\, \text{dof}_{\text{his}}} \sqrt{\mathbf{d}_i^T \hat{\Sigma}_{\text{his}} \mathbf{d}_i + \hat{\sigma}_{\text{his}}^2 (P^{-1})_{ii}}.$$
4. **Theil–Sen / Akritas–Theil–Sen slope fan (`analyze_series(..., slope_estimator="sen")`):** exported as `slope_fan_low/high` (not `trend_ci_*`), because it has zero width at each segment's median time and is **not** a pointwise CI for the trend. Each segment plots the median-pivoted slope envelope around $(t_{\text{med}},\, \hat{\alpha} + \hat{\beta}\, t_{\text{med}})$ bounded by $\hat{y}_{\text{med}} + \hat{\beta}_{\text{lo}}(t_i - t_{\text{med}})$ and $\hat{y}_{\text{med}} + \hat{\beta}_{\text{hi}}(t_i - t_{\text{med}})$ (Helsel, 2012).

5. **Censored segments (`analyze_series`, `slope_estimator="ols"`):** For segments containing censored values, `n_imputations` (default 20) proper multiple imputations are drawn from a normal censored linear regression in time fitted to the deseasonalized values and limits (parameters drawn from their asymptotic normal distribution), each imputed segment is fit by OLS, and slopes are pooled with Rubin's rules, $T = \bar{U} + (1 + 1/m) B$, with Barnard–Rubin (1999) degrees of freedom (`estimator="ols_mi"`).

## 9. Serial Correlation in Mann–Kendall / Theil–Sen

`analyze_series(..., slope_estimator="sen", mk_correction="ar1")` (default) inflates $\text{Var}(S)$ by $n/n^* = 1 + \frac{2}{n(n-1)(n-2)}\sum_{k=1}^{n-1}(n-k)(n-k-1)(n-k-2)\,\rho^S_k$, where $\rho^S_k = \tfrac{6}{\pi}\arcsin(\hat\rho^k/2)$ are the rank correlations implied by a Gaussian AR(1) process and $\hat\rho$ is the lag-1 autocorrelation of the Sen-detrended series with the small-sample correction $(n r_1 + 1)/(n - 4)$, truncated to $[0, 0.95]$. The inflation applies to both the Mann–Kendall $p$-value and the Sen CI. `mk_correction="hamed_rao"` uses the significant-lags estimate of Hamed & Rao (1998), which matches `pymannkendall` exactly but stays liberal in small samples; `"none"` assumes independence.
