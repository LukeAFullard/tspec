# Citations & Academic Attribution

`tspec` is a clean-room MIT-licensed Python package written directly from the peer-reviewed literature below. If you use `tspec` in published research, please cite the software (`CITATION.cff`) together with the relevant methodological papers.

## Primary Academic References

1. **JUST Core Algorithm (Jump Detection & Decomposition):**  
   Ghaderpour, E., & Vujadinovic, T. (2020). Change Detection within Remotely Sensed Satellite Image Time Series via Spectral Analysis. *Remote Sensing*, 12(23), 4001. https://doi.org/10.3390/rs12234001

2. **JUST Near-Real-Time Monitoring:**  
   Ghaderpour, E., & Vujadinovic, T. (2020). The Potential of the Least-Squares Spectral and Cross-Wavelet Analyses for Near-Real-Time Disturbance Detection. *Remote Sensing*, 12(15), 2446. https://doi.org/10.3390/rs12152446

3. **JUST Software Architecture:**  
   Ghaderpour, E. (2021). JUST: MATLAB and Python Software for Change Detection and Time Series Analysis. *GPS Solutions*, 25(3), 85. https://doi.org/10.1007/s10291-021-01118-x

4. **STPD Core Algorithm (Continuous Turning Points & NDRI):**  
   Ghaderpour, E., Antonielli, B., Bozzano, F., Scarascia Mugnozza, G., & Mazzanti, P. (2024). A fast and robust method for detecting trend turning points in InSAR displacement time series. *Computers & Geosciences*, 185, 105546. https://doi.org/10.1016/j.cageo.2024.105546

5. **ALLSSA Spectral Engine:**  
   Ghaderpour, E., Liao, W., & Lamoureux, M. P. (2018). Antileakage least-squares spectral analysis for seismic data regularization and random noise attenuation. *Geophysics*, 83(3), V157–V170. https://doi.org/10.1190/geo2017-0284.1

6. **LSWA / LSWAVE Foundations:**  
   - Ghaderpour, E., & Pagiatakis, S. D. (2017). Least-Squares Wavelet Analysis of Unequally Spaced and Non-stationary Time Series and Its Applications. *Mathematical Geosciences*, 49(7), 819–844. https://doi.org/10.1007/s11004-017-9691-0
   - Ghaderpour, E., & Pagiatakis, S. D. (2019). LSWAVE: a MATLAB software for the least-squares wavelet and cross-wavelet analyses. *GPS Solutions*, 23(2), 50. https://doi.org/10.1007/s10291-019-0841-3

7. **Hydrological Applications:**  
   Ghaderpour, E., Vujadinovic, T., & Hassan, Q. K. (2021). Application of the Least-Squares Wavelet software in hydrology: Athabasca River Basin. *Journal of Hydrology: Regional Studies*, 36, 100847. https://doi.org/10.1016/j.ejrh.2021.100847

8. **Censored Environmental Data & `ndimpute`:**  
   - Helsel, D. R. (2012). *Statistics for Censored Environmental Data Using Minitab and R* (2nd ed.). Wiley.
   - Turnbull, B. W. (1976). The empirical distribution function with arbitrarily grouped, censored and truncated data. *Journal of the Royal Statistical Society: Series B*, 38(3), 290–295.
   - Fullard, L. A. (2026). *ndimpute: Unbiased statistical imputation for censored data*. MIT License. https://github.com/LukeAFullard/ndimpute

9. **Sequential Alpha-Spending Boundaries (`just_monitor`):**  
   - Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials. *Biometrika*, 70(3), 659–663. https://doi.org/10.1093/biomet/70.3.659

10. **Robust M/MM-Estimation & Non-Parametric Trend Slopes (`robust="huber" | "mm"`, `slope_estimator="sen" | "l1" | "huber"`):**  
    - Huber, P. J. (1981). *Robust Statistics*. John Wiley & Sons. https://doi.org/10.1002/0471725250
    - Schrader, R. M., & Hettmansperger, T. P. (1980). Improving analysis of variance by using M-estimators. *Biometrika*, 67(1), 93–101. https://doi.org/10.1093/biomet/67.1.93
    - Yohai, V. J. (1987). High breakdown-point and high efficiency robust estimates for regression (context only: `robust="mm"` is a Huber-started MM-type estimator, not Yohai's S-started MM). *The Annals of Statistics*, 15(2), 642–656. https://doi.org/10.1214/aos/1176350366
    - Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, 63(324), 1379–1389. https://doi.org/10.1080/01621459.1968.10480934
    - Akritas, M. G., Murphy, S. A., & LaValley, M. P. (1995). The Theil-Sen estimator with doubly censored data and applications to astronomy. *Journal of the American Statistical Association*, 90(429), 170–177. https://doi.org/10.1080/01621459.1995.10476499
    - Gilbert, R. O. (1987). *Statistical Methods for Environmental Pollution Monitoring*. Van Nostrand Reinhold.
    - He, X., Jurečková, J., Koenker, R., & Portnoy, S. (1990). Tail behavior of regression estimators and their breakdown points. *Econometrica*, 58(5), 1195–1214. https://doi.org/10.2307/2938306

11. **Calibrated Inference (split search, multiplicity, bias correction, autocorrelation):**  
    - Andrews, D. W. K. (1993). Tests for parameter instability and structural change with unknown change point. *Econometrica*, 61(4), 821–856. https://doi.org/10.2307/2951764 *(sup-F statistic; `just_jumps(jump_calibration=...)`)*
    - Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. *(Holm adjustment across candidate jumps)*
    - Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the effect of bias estimation on coverage accuracy in nonparametric inference. *Journal of the American Statistical Association*, 113(522), 767–779. https://doi.org/10.1080/01621459.2017.1285776 *(bias-corrected `just_decompose` trend band)*
    - Hamed, K. H., & Rao, A. R. (1998). A modified Mann–Kendall trend test for autocorrelated data. *Journal of Hydrology*, 204, 182–196. https://doi.org/10.1016/S0022-1694(97)00125-X *(`mk_correction`)*
    - Marriott, F. H. C., & Pope, J. A. (1954). Bias in the estimation of autocorrelations. *Biometrika*, 41(3/4), 390–402. https://doi.org/10.1093/biomet/41.3-4.390 *(small-sample correction in `mk_correction="ar1"`)*

12. **Censored Regression & Multiple Imputation:**  
    - Tobin, J. (1958). Estimation of relationships for limited dependent variables. *Econometrica*, 26(1), 24–36. https://doi.org/10.2307/1907382 *(censored-regression imputation model)*
    - Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley. https://doi.org/10.1002/9780470316696
    - Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of freedom with multiple imputation. *Biometrika*, 86(4), 948–955. https://doi.org/10.1093/biomet/86.4.948
    - Meng, X.-L. (1994). Multiple-imputation inferences with uncongenial sources of input. *Statistical Science*, 9(4), 538–558. https://doi.org/10.1214/ss/1177010269
