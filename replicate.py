"""Replication code: 'Underwriting Cycles and the Calibration of Premium Risk
under Solvency II'.

Reproduces every estimate reported in the manuscript:
  Sec. 5.2  capital comparison and variance decomposition (Table 1)
  Sec. 5.3  AR(2) representation; two-regime Markov-switching estimates,
            transition probabilities, profile-likelihood region and model
            comparison (Table 2, panels A-C)
  Sec. 5.4  bootstrap confidence intervals (Table 3)
  Sec. 5.5  accident-year estimates and the Lloyd's series (Table 4, Fig. 3)
  Sec. 5.6  reinsurance decomposition (Table 5)
  Sec. 6.2  simulated coverage study (Table 6); capital-neutral calibration
            of the loading and the level diagnostic (Table 7)
  Sec. 6.3  real-time checks: the price-index rule (8) applied to the BLS
            producer price index, and the regime model (Table 8)
  App. A    distributional sensitivity (Table A1)
  Figures 1, 2 and 3; Fig. S1 (supplementary)

The Markov-switching likelihood with regime-dependent variances is unbounded
(Kiefer 1978); estimation therefore uses EM under Hathaway's (1985) constraint
on the ratio of regime standard deviations, followed by a direct polish of the
log-likelihood. No statsmodels dependency.

Requirements: Python >= 3.10, numpy, scipy, pandas, matplotlib.
Run:  python replicate.py            (full run, about 8 minutes)
      python replicate.py --quick    (point estimates and tables without
                                      bootstrap, profile and simulation)
Reads data.csv next to this script; writes figures and CSV files to results/.
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUICK = "--quick" in sys.argv
np.seterr(over="ignore")
SEED = 20260915
Z = stats.norm.ppf(0.995)
SD_RATIO_CAP = 10.0            # Hathaway (1985) constraint: max sd / min sd
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7})
T0 = time.time()

def banner(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(os.path.join(HERE, "data.csv"))
cy = df.dropna(subset=["cr_cy"])
y, yrs = cy.cr_cy.values.astype(float), cy.year.values.astype(int)
ay_df = df.dropna(subset=["cr_ay"])
ya, yrs_a = ay_df.cr_ay.values.astype(float), ay_df.year.values.astype(int)
banner("DATA")
print(f"calendar year: n = {len(y)}, {yrs[0]}-{yrs[-1]}, mean {y.mean():.2f}, sd {y.std(ddof=1):.3f}")
print(f"accident year: n = {len(ya)}, {yrs_a[0]}-{yrs_a[-1]}, mean {ya.mean():.2f}, sd {ya.std(ddof=1):.3f}")
m = np.isin(yrs, yrs_a)
print(f"corr(CY, AY) on {yrs_a[0]}-{yrs_a[-1]}: {np.corrcoef(y[m], ya)[0,1]:.3f}")
print(f"corr(CY_t, AY_t-2): {np.corrcoef(y[m][2:], ya[:-2])[0,1]:.3f}")

# --------------------------------------------------------------------------
# AR(2) by OLS (Sec. 5.3)
# --------------------------------------------------------------------------
banner("AR(2), Cummins-Outreville tradition")
X = np.column_stack([np.ones(len(y) - 2), y[1:-1], y[:-2]])
c, a1, a2 = np.linalg.lstsq(X, y[2:], rcond=None)[0]
print(f"const {c:.1f}  a1 {a1:.3f}  a2 {a2:.3f}")
if a1**2 + 4 * a2 < 0:
    print(f"complex roots; implied cycle period {2*np.pi/np.arccos(a1/(2*np.sqrt(-a2))):.1f} years")

# --------------------------------------------------------------------------
# Markov-switching machinery
# --------------------------------------------------------------------------
def ergodic(P):
    a, b = P[0, 1], P[1, 0]
    if P.shape[0] == 2:
        return np.array([b / (a + b), a / (a + b)]) if a + b > 0 else np.array([.5, .5])
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))]); pi = np.abs(pi)
    return pi / pi.sum()

def filt_smooth(x, P, mu, sd):
    """Hamilton (1989) filter and Kim (1994) smoother. Returns filtered,
    predicted, smoothed probabilities, pairwise smoothed joints, log-likelihood."""
    T, K = len(x), len(mu)
    eta = stats.norm.pdf(x[:, None], mu[None, :], sd[None, :])
    filt = np.zeros((T, K)); pred = np.zeros((T, K)); ll = 0.0
    prev = ergodic(P)
    for t in range(T):
        pr = prev if t == 0 else prev @ P
        pred[t] = pr
        num = pr * eta[t]; den = num.sum()
        if not np.isfinite(den) or den <= 0:
            return None, None, None, None, -np.inf
        ll += np.log(den); filt[t] = num / den; prev = filt[t]
    sm = np.zeros((T, K)); sm[-1] = filt[-1]; joint = np.zeros((T - 1, K, K))
    for t in range(T - 2, -1, -1):
        ratio = sm[t + 1] / np.where(pred[t + 1] > 0, pred[t + 1], 1e-300)
        j = filt[t][:, None] * P * ratio[None, :]
        joint[t] = j; sm[t] = j.sum(1); sm[t] /= sm[t].sum()
    return filt, pred, sm, joint, ll

def cap_ratio(sd, cap=SD_RATIO_CAP):
    lo, hi = sd.min(), sd.max()
    if hi / lo <= cap:
        return sd
    g = np.exp(np.log(sd).mean()); a = np.log(cap) / np.log(hi / lo)
    return g * np.exp(a * (np.log(sd) - np.log(g)))

def em(x, K=2, switching_var=True, n_starts=12, maxit=400, seed=0, tol=1e-9, do_polish=True):
    rng = np.random.default_rng(seed); best = None
    for s in range(n_starts):
        if s == 0:
            mu = np.quantile(x, np.linspace(0.85, 0.25, K)); sd = np.full(K, 0.8 * x.std(ddof=1))
        else:
            mu = rng.choice(x, K, replace=False) + rng.normal(0, 1.5, K)
            sd = np.abs(rng.uniform(0.4, 1.4, K) * x.std(ddof=1))
        if not switching_var:
            sd = np.full(K, sd.mean())
        P = np.full((K, K), 0.12 / (K - 1)); np.fill_diagonal(P, 0.88); ll_old = -np.inf
        for it in range(maxit):
            filt, pred, sm, joint, ll = filt_smooth(x, P, mu, sd)
            if not np.isfinite(ll):
                break
            w = sm.sum(0)
            if (w < 1e-8).any():
                break
            mu = (sm * x[:, None]).sum(0) / w
            if switching_var:
                sd = cap_ratio(np.maximum(np.sqrt((sm * (x[:, None] - mu) ** 2).sum(0) / w), 1e-3))
            else:
                sd = np.full(K, np.sqrt((sm * (x[:, None] - mu) ** 2).sum() / len(x)))
            num = joint.sum(0); rs = num.sum(1, keepdims=True)
            if (rs <= 0).any():
                break
            P = np.clip(num / rs, 1e-4, 1 - 1e-4); P /= P.sum(1, keepdims=True)
            if ll - ll_old < tol and it > 8:
                break
            ll_old = ll
        *_, ll = filt_smooth(x, P, mu, sd)
        if np.isfinite(ll) and (best is None or ll > best["ll"] + 1e-9):
            best = dict(ll=ll, mu=mu.copy(), sd=sd.copy(), P=P.copy())
    return polish(x, best, switching_var, do_polish) if best else None

def _pack(mu, sd, P, switching_var):
    K = len(mu)
    th = [np.log(P[i, i] / (1 - P[i, i])) for i in range(K)] if K == 2 else list(np.log(P[:, :-1] / P[:, -1:]).ravel())
    th += list(mu)
    th += list(np.log(sd)) if switching_var else [np.log(sd[0])]
    return np.array(th)

def _unpack(th, K, switching_var):
    if K == 2:
        p00, p11 = 1 / (1 + np.exp(-th[0])), 1 / (1 + np.exp(-th[1]))
        P = np.array([[p00, 1 - p00], [1 - p11, p11]]); i = 2
    else:
        L = th[:K * (K - 1)].reshape(K, K - 1); E = np.exp(np.column_stack([L, np.zeros(K)]))
        P = E / E.sum(1, keepdims=True); i = K * (K - 1)
    mu = th[i:i + K]; i += K
    sd = np.exp(th[i:i + K]) if switching_var else np.full(K, np.exp(th[i]))
    return mu, sd, P

def polish(x, est, switching_var, do_polish=True):
    """Direct maximisation of the log-likelihood from the EM solution,
    retaining the ratio constraint. With do_polish=False the EM solution is
    returned in the same format (used inside the bootstrap and backtest)."""
    K = len(est["mu"])
    def negll(th):
        mu, sd, P = _unpack(th, K, switching_var)
        if sd.max() / sd.min() > SD_RATIO_CAP:
            return 1e6
        return -filt_smooth(x, P, mu, sd)[4]
    mu, sd, P, ll = est["mu"], est["sd"], est["P"], est["ll"]
    if do_polish:
        th0 = _pack(mu, sd, P, switching_var)
        r = optimize.minimize(negll, th0, method="Nelder-Mead",
                              options=dict(maxiter=20000, xatol=1e-8, fatol=1e-10))
        if np.isfinite(r.fun) and -r.fun > ll:
            mu, sd, P = _unpack(r.x, K, switching_var); ll = -r.fun
    o = np.argsort(-mu)                                  # regime 0 = soft (highest mean)
    mu, sd, P = mu[o], sd[o], P[np.ix_(o, o)]
    filt, pred, sm, _, ll = filt_smooth(x, P, mu, sd)
    k = K * (K - 1) + K + (K if switching_var else 1)
    return dict(ll=ll, mu=mu, sd=sd, P=P, smoothed=sm, filtered=filt, k=k,
                aic=-2 * ll + 2 * k, bic=-2 * ll + np.log(len(x)) * k, ergodic=ergodic(P))

def capital(mu, sd):
    """Break-even basis and unexpected-loss basis, per 100 of premium."""
    return np.maximum(mu - 100.0, 0.0) + Z * sd, Z * sd

# --------------------------------------------------------------------------
# Main estimates (Sec. 5.3, Table 2 panel A)
# --------------------------------------------------------------------------
banner("TWO-REGIME MARKOV SWITCHING, CALENDAR YEAR (Table 2, panel A)")
ms = em(y, K=2, switching_var=True, n_starts=40, seed=SEED)
print(f"log L {ms['ll']:.3f}   AIC {ms['aic']:.2f}   BIC {ms['bic']:.2f}   k = {ms['k']}")
print(f"soft:  mu {ms['mu'][0]:.2f}  sd {ms['sd'][0]:.3f}")
print(f"hard:  mu {ms['mu'][1]:.2f}  sd {ms['sd'][1]:.3f}")
print(f"p_SS {ms['P'][0,0]:.3f}  p_HH {ms['P'][1,1]:.3f}   implied durations "
      f"{1/(1-ms['P'][0,0]):.1f}, {1/(1-ms['P'][1,1]):.1f} years   ergodic P(soft) {ms['ergodic'][0]:.3f}")
soft_years = yrs[ms["smoothed"][:, 0] > 0.5]
print("years with smoothed P(soft) > 0.5:", [int(v) for v in soft_years])
print("smoothed P(soft):", " ".join(f"{yr}:{p:.2f}" for yr, p in zip(yrs, ms["smoothed"][:, 0])))
mu_u, sd_u = y.mean(), y.std(ddof=1)

# unboundedness illustration
sp_mu = np.array([116.9, y[y != 116.9].mean()]); sp_sd = np.array([1e-3, y[y != 116.9].std(ddof=0)])
*_, ll_spike = filt_smooth(y, np.array([[0.001, 0.999], [0.0357, 0.9643]]), sp_mu, sp_sd)
print(f"degenerate spike (2001 alone, sd -> 0): log L {ll_spike:.1f} -> unbounded; constrained interior maximum used")

# --------------------------------------------------------------------------
# Model comparison (Table 2 panel C)
# --------------------------------------------------------------------------
banner("MODEL COMPARISON, CALENDAR YEAR (Table 2, panel C)")
ll1 = stats.norm.logpdf(y, y.mean(), y.std(ddof=0)).sum()
rows = [("one regime (i.i.d. normal)", ll1, 2, -2 * ll1 + 4, -2 * ll1 + np.log(len(y)) * 2, f"mu {y.mean():.2f}, sd {y.std(ddof=0):.2f}")]
ms_cv = em(y, K=2, switching_var=False, n_starts=40, seed=SEED)
rows.append(("two regimes, common variance", ms_cv["ll"], ms_cv["k"], ms_cv["aic"], ms_cv["bic"],
             f"mu {ms_cv['mu'][0]:.2f}/{ms_cv['mu'][1]:.2f}, sd {ms_cv['sd'][0]:.2f}"))
rows.append(("two regimes, switching variance", ms["ll"], ms["k"], ms["aic"], ms["bic"],
             f"mu {ms['mu'][0]:.2f}/{ms['mu'][1]:.2f}, sd {ms['sd'][0]:.2f}/{ms['sd'][1]:.2f}"))
ms3 = em(y, K=3, switching_var=True, n_starts=40, seed=SEED)
if ms3:
    rows.append(("three regimes, switching variance", ms3["ll"], ms3["k"], ms3["aic"], ms3["bic"],
                 "mu " + "/".join(f"{v:.1f}" for v in ms3["mu"]) + ", sd " + "/".join(f"{v:.2f}" for v in ms3["sd"])))
print(f"{'specification':36s}{'log L':>9s}{'k':>4s}{'AIC':>9s}{'BIC':>9s}   estimates")
for r in rows:
    print(f"{r[0]:36s}{r[1]:9.2f}{r[2]:4d}{r[3]:9.2f}{r[4]:9.2f}   {r[5]}")
LR = 2 * (ms["ll"] - ms_cv["ll"])
print(f"LR, common vs switching variance: {LR:.2f} on 1 df, p = {1-stats.chi2.cdf(LR,1):.3f}")
be_cv, ul_cv = capital(ms_cv["mu"], ms_cv["sd"])
print(f"common-variance capital: break-even soft {be_cv[0]:.2f} hard {be_cv[1]:.2f}; unexpected-loss both {ul_cv[0]:.2f}")

# --------------------------------------------------------------------------
# Table 1: capital comparison and variance decomposition (Sec. 5.2)
# --------------------------------------------------------------------------
banner("CAPITAL AT THE 99.5% STANDARD (Table 1)")
be_s, ul_s = capital(np.array([mu_u]), np.array([sd_u]))
be, ul = capital(ms["mu"], ms["sd"])
print(f"{'state':28s}{'mean':>8s}{'sd':>7s}{'UL z*sd':>9s}{'break-even':>12s}{'ratio BE':>10s}{'ratio UL':>10s}")
print(f"{'unconditional (static)':28s}{mu_u:8.2f}{sd_u:7.3f}{ul_s[0]:9.2f}{be_s[0]:12.2f}{1.0:10.2f}{1.0:10.2f}")
for i, lab in enumerate(["hard/moderate regime", "soft regime"][::-1]):
    j = 0 if lab.startswith("soft") else 1
    print(f"{lab:28s}{ms['mu'][j]:8.2f}{ms['sd'][j]:7.3f}{ul[j]:9.2f}{be[j]:12.2f}{be[j]/be_s[0]:10.3f}{ul[j]/ul_s[0]:10.3f}")
print(f"3-sigma rule on the sample coefficient of variation: {300*sd_u/mu_u:.2f}")
print(f"static as % of soft-state break-even requirement: {100*be_s[0]/be[0]:.1f}% (shortfall {100*(1-be_s[0]/be[0]):.1f}%)")
print(f"static as % of hard-state requirement: {100*be_s[0]/be[1]:.0f}%")
w = ms["smoothed"][:, 0].mean()
between = w * (1 - w) * (ms["mu"][0] - ms["mu"][1]) ** 2
within = w * ms["sd"][0] ** 2 + (1 - w) * ms["sd"][1] ** 2
print(f"variance decomposition at smoothed frequencies (w = {w:.3f}): within {within:.1f} + between {between:.1f} "
      f"= {within+between:.1f}; between share {100*between/(within+between):.0f}%")

# --------------------------------------------------------------------------
# Appendix A: distributional sensitivity (Table A1)
# --------------------------------------------------------------------------
banner("DISTRIBUTIONAL SENSITIVITY AT 99.5% (Table A1)")
def q_lognormal(mu, sd):
    v = np.log1p((sd / mu) ** 2); return np.exp(np.log(mu) - v / 2 + stats.norm.ppf(0.995) * np.sqrt(v))
def q_t(mu, sd, nu):
    return mu + sd / np.sqrt(nu / (nu - 2)) * stats.t.ppf(0.995, nu)
sig = 0.10
f_ln = np.exp(Z * np.sqrt(np.log(1 + sig**2))) / np.sqrt(1 + sig**2) - 1
print(f"lognormal quantile factor f(sigma)/sigma at sigma=0.10: {f_ln/sig:.2f};", end=" ")
sig = 0.17; f_ln = np.exp(Z * np.sqrt(np.log(1 + sig**2))) / np.sqrt(1 + sig**2) - 1
print(f"at sigma=0.17: {f_ln/sig:.2f}")
specs = [("Gaussian", lambda m, s: m + Z * s), ("lognormal", q_lognormal)] + \
        [(f"Student t({nu})", (lambda nu: (lambda m, s: q_t(m, s, nu)))(nu)) for nu in (10, 7, 5, 4)]
print(f"{'distribution':16s}{'soft BE':>9s}{'hard BE':>9s}{'static BE':>11s}{'soft/static':>12s}{'soft UL':>9s}{'hard UL':>9s}")
for name, q in specs:
    qs, qh, qu = q(ms["mu"][0], ms["sd"][0]), q(ms["mu"][1], ms["sd"][1]), q(mu_u, sd_u)
    s_be, h_be, u_be = qs - 100, max(qh - 100, qh - ms["mu"][1]), qu - 100
    print(f"{name:16s}{s_be:9.2f}{h_be:9.2f}{u_be:11.2f}{s_be/u_be:12.3f}{qs-ms['mu'][0]:9.2f}{qh-ms['mu'][1]:9.2f}")

# --------------------------------------------------------------------------
# Accident-year estimates (Sec. 5.5, Table 4)
# --------------------------------------------------------------------------
banner("ACCIDENT-YEAR ESTIMATES (Table 4)")
msa = em(ya, K=2, switching_var=True, n_starts=40, seed=SEED)
be_a, ul_a = capital(msa["mu"], msa["sd"]); be_as, ul_as = capital(np.array([ya.mean()]), np.array([ya.std(ddof=1)]))
print(f"log L {msa['ll']:.3f}; soft mu {msa['mu'][0]:.2f} sd {msa['sd'][0]:.3f}; hard mu {msa['mu'][1]:.2f} sd {msa['sd'][1]:.3f}")
print(f"p_SS {msa['P'][0,0]:.3f} p_HH {msa['P'][1,1]:.3f}; ergodic P(soft) {msa['ergodic'][0]:.3f}")
print("years with smoothed P(soft) > 0.5:", [int(v) for v in yrs_a[msa["smoothed"][:, 0] > 0.5]])
print(f"regime mean spread: CY {ms['mu'][0]-ms['mu'][1]:.2f}, AY {msa['mu'][0]-msa['mu'][1]:.2f}")
print(f"AY capital: static BE {be_as[0]:.2f}, soft BE {be_a[0]:.2f}, hard BE {be_a[1]:.2f}; soft/static {be_a[0]/be_as[0]:.3f}")
print(f"AY capital UL: static {ul_as[0]:.2f}, soft {ul_a[0]:.2f}, hard {ul_a[1]:.2f}")
msc = em(y[m], K=2, switching_var=True, n_starts=40, seed=SEED)
print(f"CY on the same 1996-2015 window: soft mu {msc['mu'][0]:.2f} sd {msc['sd'][0]:.2f}; hard mu {msc['mu'][1]:.2f} sd {msc['sd'][1]:.2f}; spread {msc['mu'][0]-msc['mu'][1]:.2f}")


# --------------------------------------------------------------------------
# Second market: Lloyd's (Sec. 5.5, Table 4 rows, Fig. 3)
# --------------------------------------------------------------------------
banner("LLOYD'S OF LONDON: CALENDAR-YEAR AND ATTRITIONAL ACCIDENT-YEAR SERIES (Table 4, Fig. 3)")
lc = df.dropna(subset=["lloyds_cr_cy"]); ylc, yrs_lc = lc.lloyds_cr_cy.values.astype(float), lc.year.values.astype(int)
la = df.dropna(subset=["lloyds_ay_exmajor"]); yla, yrs_la = la.lloyds_ay_exmajor.values.astype(float), la.year.values.astype(int)
print(f"calendar-year combined ratio {yrs_lc[0]}-{yrs_lc[-1]}: n = {len(ylc)}, mean {ylc.mean():.2f}, sd {ylc.std(ddof=1):.2f}")
print(f"accident-year ratio excl. major claims {yrs_la[0]}-{yrs_la[-1]}: n = {len(yla)}, mean {yla.mean():.2f}, sd {yla.std(ddof=1):.2f}")
LL = {}
for lab, yv, yr in [("calendar year", ylc, yrs_lc), ("attritional accident year", yla, yrs_la)]:
    r = em(yv, K=2, switching_var=True, n_starts=60, seed=SEED); rc = em(yv, K=2, switching_var=False, n_starts=60, seed=SEED)
    ll1 = stats.norm.logpdf(yv, yv.mean(), yv.std(ddof=0)).sum()
    be_, ul_ = capital(r["mu"], r["sd"]); sbe_, sul_ = capital(np.array([yv.mean()]), np.array([yv.std(ddof=1)]))
    w_ = r["smoothed"][:, 0].mean(); btw = w_ * (1 - w_) * (r["mu"][0] - r["mu"][1]) ** 2; wth = w_ * r["sd"][0] ** 2 + (1 - w_) * r["sd"][1] ** 2
    LL[lab] = r
    print(f"\n{lab}: switching variance log L {r['ll']:.2f} (BIC {r['bic']:.2f}); common variance log L {rc['ll']:.2f} (BIC {rc['bic']:.2f}); one regime BIC {-2*ll1+np.log(len(yv))*2:.2f}; LR {2*(r['ll']-rc['ll']):.2f}")
    print(f"  soft mu {r['mu'][0]:.2f} sd {r['sd'][0]:.2f} | hard/moderate mu {r['mu'][1]:.2f} sd {r['sd'][1]:.2f} | spread {r['mu'][0]-r['mu'][1]:.2f} | common-variance means {rc['mu'][0]:.2f}/{rc['mu'][1]:.2f} sd {rc['sd'][0]:.2f}")
    print(f"  p_SS {r['P'][0,0]:.3f} p_HH {r['P'][1,1]:.3f}; soft years {[int(v) for v in yr[r['smoothed'][:,0]>0.5]]}")
    print(f"  variance decomposition at smoothed frequencies (w = {w_:.3f}): within {wth:.1f} + between {btw:.1f} = {wth+btw:.1f}; between share {100*btw/(wth+btw):.0f}%")
    print(f"  capital, unexpected-loss basis: static {sul_[0]:.2f}, soft {ul_[0]:.2f}, hard/moderate {ul_[1]:.2f}; soft/static {ul_[0]/sul_[0]:.3f}, hard/static {ul_[1]/sul_[0]:.3f}")
# stationary block bootstrap of the attritional series (spread and its sign)
rng_l = np.random.default_rng(SEED); sp = []; sdiff = []
for b in range(1000 if not QUICK else 50):
    xb = np.empty(len(yla)); i = 0
    while i < len(yla):
        L = rng_l.geometric(0.2); st = rng_l.integers(0, len(yla))
        for k in range(L):
            if i >= len(yla): break
            xb[i] = yla[(st + k) % len(yla)]; i += 1
    rb = em(xb, K=2, switching_var=True, n_starts=4, seed=b, do_polish=False)
    if rb is not None: sp.append(rb["mu"][0] - rb["mu"][1]); sdiff.append(rb["sd"][0] - rb["sd"][1])
sp = np.array(sp); sdiff = np.array(sdiff)
print(f"\nstationary block bootstrap of the attritional series (B = {len(sp)}): spread median {np.median(sp):.2f}, 95% [{np.quantile(sp,.025):.2f}, {np.quantile(sp,.975):.2f}], P(spread > 2) = {(sp>2).mean():.3f}; "
      f"sd(S) - sd(H) median {np.median(sdiff):.2f}, 95% [{np.quantile(sdiff,.025):.2f}, {np.quantile(sdiff,.975):.2f}]")
# Figure 3
r_la = LL["attritional accident year"]
fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)
ax.fill_between(yrs_la, 78, 118, where=r_la["smoothed"][:, 0] > 0.5, color="0.85", label="Soft regime, attritional model (smoothed $P>0.5$)")
ax.plot(yrs_lc, ylc, color="0.55", lw=1.1, marker="o", ms=2.4, label="Calendar-year combined ratio")
ax.plot(yrs_la, yla, color="black", lw=1.4, marker="s", ms=2.6, label="Accident-year ratio excluding major claims")
ax.axhline(100, color="0.4", lw=0.8, ls="--")
ax.set_xlim(2001.4, 2026.2); ax.set_ylim(78, 118); ax.set_ylabel("Ratio (%)"); ax.set_xlabel("Year")
ax.legend(loc="upper left", fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_lloyds.png")); plt.close()
print("figure written: fig3_lloyds.png")

# --------------------------------------------------------------------------
# Reinsurance decomposition (Sec. 5.6, Table 5)
# --------------------------------------------------------------------------
banner("REINSURANCE DECOMPOSITION (Table 5)")
print(f"channel A, break-even basis: static {be_s[0]:.2f} vs soft {be[0]:.2f}: static is {100*be_s[0]/be[0]:.1f}% of requirement")
print(f"channel A, unexpected-loss basis: static {ul_s[0]:.2f} vs soft {ul[0]:.2f}: static is {100*ul_s[0]/ul[0]:.1f}% of requirement")
print(f"{'scenario':44s}{'volume':>8s}{'BE gap':>9s}{'UL gap':>9s}")
for lab, d, g, yr1 in [("index -16%, limit unchanged, year 2 onward", .16, 0, False),
                       ("index -16%, limit +5%, year 2 onward", .16, .05, False),
                       ("index -16%, limit +10%, year 2 onward", .16, .10, False),
                       ("index -12%, limit +5%, year 2 onward", .12, .05, False),
                       ("any of the above, year 1 (Art. 116(3) floor)", .16, 0, True)]:
    vol = 0.0 if yr1 else (1 - d) * (1 + g) - 1
    print(f"{lab:44s}{100*vol:8.1f}{100*(1-(be_s[0]/be[0])*(1+vol)):9.1f}{100*(1-(ul_s[0]/ul[0])*(1+vol)):9.1f}")

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)
ax.fill_between(yrs, 88, 120, where=ms["smoothed"][:, 0] > 0.5, color="0.85", label="Soft-market regime (smoothed $P>0.5$)")
ax.plot(yrs, y, color="black", lw=1.4, marker="o", ms=2.8)
ax.axhline(100, color="0.4", lw=0.8, ls="--"); ax.axhline(mu_u, color="0.4", lw=0.8, ls=":")
ax.annotate("break-even", xy=(2024.8, 100.2), fontsize=7.5, color="0.3")
ax.annotate(f"sample mean {mu_u:.1f}", xy=(2013.7, 102.0), fontsize=7.5, color="0.3")
for yr, v, dy in [(2001, 116.9, 1.0), (2006, 93.3, -2.6), (2017, 103.9, 1.2), (2025, 92.9, -2.6)]:
    ax.annotate(f"{yr}: {v:.1f}", xy=(yr, v), xytext=(yr - 1.2, v + dy), fontsize=7.5)
ax.set_xlim(1995.4, 2026.2); ax.set_ylim(88, 120)
ax.set_ylabel("Combined ratio (%)"); ax.set_xlabel("Calendar year")
ax.legend(loc="upper right", fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_regimes.png")); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)
ax.fill_between(yrs_a, 88, 120, where=msa["smoothed"][:, 0] > 0.5, color="0.85", label="Soft regime, accident-year model (smoothed $P>0.5$)")
ax.plot(yrs, y, color="0.55", lw=1.1, marker="o", ms=2.4, label="Calendar year")
ax.plot(yrs_a, ya, color="black", lw=1.4, marker="s", ms=2.6, label="Accident year (2015 evaluation)")
ax.axhline(100, color="0.4", lw=0.8, ls="--")
ax.set_xlim(1995.4, 2026.2); ax.set_ylim(88, 120)
ax.set_ylabel("Combined ratio (%)"); ax.set_xlabel("Year")
ax.legend(loc="upper right", fontsize=7.5, frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_ay_cy.png")); plt.close()
print("\nfigures written: fig1_regimes.png, fig2_ay_cy.png")

if QUICK:
    print(f"\n--quick run complete ({time.time()-T0:.0f}s)"); sys.exit()

# --------------------------------------------------------------------------
# Profile likelihood over the transition matrix (Sec. 5.3)
# --------------------------------------------------------------------------
banner("PROFILE LIKELIHOOD OVER (p_SS, p_HH) (Table 2, panel B)")
def em_fixedP(x, P, n_starts=4, maxit=250, seed=1):
    rng = np.random.default_rng(seed); best = None
    for s in range(n_starts):
        mu = ms["mu"].copy() if s == 0 else rng.choice(x, 2, replace=False) + rng.normal(0, 1.5, 2)
        sd = ms["sd"].copy() if s == 0 else np.abs(rng.uniform(0.5, 1.3, 2) * x.std(ddof=1))
        for it in range(maxit):
            filt, pred, sm, joint, ll = filt_smooth(x, P, mu, sd)
            if not np.isfinite(ll):
                break
            w = sm.sum(0); mu = (sm * x[:, None]).sum(0) / w
            sd = cap_ratio(np.maximum(np.sqrt((sm * (x[:, None] - mu) ** 2).sum(0) / w), 1e-3))
        *_, ll = filt_smooth(x, P, mu, sd)
        if np.isfinite(ll) and (best is None or ll > best[0]):
            best = (ll, mu.copy(), sd.copy())
    return best
grid = np.round(np.arange(0.60, 0.991, 0.03), 3)
prof = np.full((len(grid), len(grid)), -np.inf); mom = {}
for i, pss in enumerate(grid):
    for j, phh in enumerate(grid):
        r = em_fixedP(y, np.array([[pss, 1 - pss], [1 - phh, phh]]))
        if r:
            prof[i, j] = r[0]; mom[(i, j)] = (r[1], r[2])
imax = np.unravel_index(np.argmax(prof), prof.shape)
cut = prof.max() - stats.chi2.ppf(0.95, 2) / 2
inreg = prof >= cut
ii, jj = np.where(inreg)
print(f"grid maximum log L {prof.max():.2f} at p_SS {grid[imax[0]]:.2f}, p_HH {grid[imax[1]]:.2f}")
print(f"95% profile region: p_SS in [{grid[ii].min():.2f}, {grid[ii].max():.2f}], p_HH in [{grid[jj].min():.2f}, {grid[jj].max():.2f}]")
print(f"implied durations in region: soft {1/(1-grid[ii].min()):.1f}-{1/(1-grid[ii].max()):.0f} yrs, hard {1/(1-grid[jj].min()):.1f}-{1/(1-grid[jj].max()):.0f} yrs")
mus = np.array([mom[(a, b)][0] for a, b in zip(ii, jj)]); sds = np.array([mom[(a, b)][1] for a, b in zip(ii, jj)])
o = np.argsort(-mus, axis=1); mus = np.take_along_axis(mus, o, 1); sds = np.take_along_axis(sds, o, 1)
print(f"conditional moments across the region: mu(S) [{mus[:,0].min():.2f}, {mus[:,0].max():.2f}], mu(H) [{mus[:,1].min():.2f}, {mus[:,1].max():.2f}], "
      f"sd(S) [{sds[:,0].min():.2f}, {sds[:,0].max():.2f}], sd(H) [{sds[:,1].min():.2f}, {sds[:,1].max():.2f}]")
print(f"({time.time()-T0:.0f}s)")

# --------------------------------------------------------------------------
# Bootstrap (Sec. 5.4, Table 3)
# --------------------------------------------------------------------------
banner("BOOTSTRAP (Table 3)")
def simulate(P, mu, sd, T, rng, pi0=None):
    pi0 = ergodic(P) if pi0 is None else pi0
    s = np.empty(T, int); s[0] = rng.choice(2, p=pi0)
    for t in range(1, T):
        s[t] = rng.choice(2, p=P[s[t - 1]])
    return mu[s] + sd[s] * rng.standard_normal(T), s
def bootstrap(kind, B, seed):
    rng = np.random.default_rng(seed); out = []
    for b in range(B):
        if kind == "parametric":
            xb, _ = simulate(ms["P"], ms["mu"], ms["sd"], len(y), rng)
        elif kind == "parametric, start soft":
            xb, _ = simulate(ms["P"], ms["mu"], ms["sd"], len(y), rng, pi0=np.array([1.0, 0.0]))
        else:                                   # stationary bootstrap, Politis-Romano (1994), mean block 5
            xb = np.empty(len(y)); i = 0
            while i < len(y):
                L = rng.geometric(0.2); st = rng.integers(0, len(y))
                for k in range(L):
                    if i >= len(y):
                        break
                    xb[i] = y[(st + k) % len(y)]; i += 1
        r = em(xb, K=2, switching_var=True, n_starts=4, seed=b, do_polish=False)
        if r is None:
            continue
        bb, uu = capital(r["mu"], r["sd"]); sb, su = capital(np.array([xb.mean()]), np.array([xb.std(ddof=1)]))
        out.append(dict(mu_s=r["mu"][0], mu_h=r["mu"][1], sd_s=r["sd"][0], sd_h=r["sd"][1], spread=r["mu"][0] - r["mu"][1],
                        be_s=bb[0], be_h=bb[1], ul_s=uu[0], ratio_be_s=bb[0] / sb[0], ratio_be_h=bb[1] / sb[0],
                        ratio_ul_s=uu[0] / su[0], sign_be=float(bb[0] > sb[0]), sign_ul=float(uu[0] > su[0])))
    return pd.DataFrame(out)
B = 1000
for kind in ["parametric", "parametric, start soft", "stationary block"]:
    d = bootstrap(kind, B, SEED)
    print(f"\n{kind} (B = {len(d)}):")
    for cname in ["mu_s", "mu_h", "sd_s", "sd_h", "spread", "be_s", "be_h", "ratio_be_s", "ratio_be_h", "ratio_ul_s"]:
        v = d[cname].values
        print(f"  {cname:12s} median {np.median(v):8.3f}   95% [{np.quantile(v,.025):8.3f}, {np.quantile(v,.975):8.3f}]")
    print(f"  P(soft break-even requirement > static) = {d.sign_be.mean():.3f};  P(soft UL requirement > static UL) = {d.sign_ul.mean():.3f}")
    print(f"  ({time.time()-T0:.0f}s)")

# --------------------------------------------------------------------------
# Coverage study (Sec. 6.2, Table 6)
# --------------------------------------------------------------------------
banner("SIMULATED COVERAGE STUDY (Table 6)")
mH, sC, pH = ms["mu"][1], ms["sd"][1], 100.0
pS = pH * ms["mu"][1] / ms["mu"][0]                       # price implied by the regime means
p_state = np.array([pS, pH]); Psim = np.array([[0.92, 0.08], [0.05, 0.95]]); pi = ergodic(Psim)
mu_st, sd_st = 100 * mH / p_state, 100 * sC / p_state
print(f"price soft {pS:.2f} vs hard {pH:.0f} (softening {100*(1-pS/pH):.1f}%); implied mu(S) {mu_st[0]:.2f}, sd(S) {sd_st[0]:.2f}; ergodic P(soft) {pi[0]:.3f}")
rng = np.random.default_rng(SEED); Tn = 1_000_000
s = np.empty(Tn, np.int8); s[0] = 0 if rng.random() < pi[0] else 1; u = rng.random(Tn)
for t in range(1, Tn):
    s[t] = 0 if u[t] < Psim[s[t - 1], 0] else 1
C = mH + sC * rng.standard_normal(Tn); p = p_state[s]; CR = 100 * C / p
sigma_hat = CR.std(ddof=1) / 100; pbar = float(np.exp(np.log(p_state) @ pi))
print(f"simulated CR mean {CR.mean():.2f} sd {CR.std(ddof=1):.2f}; sigma_hat {sigma_hat:.4f}; reference price {pbar:.2f}")
def cov(charge, b_cr):
    exc = (C - b_cr * p / 100) > charge
    return 100 * exc.mean(), 100 * exc[s == 0].mean(), 100 * exc[s == 1].mean(), float(np.mean(charge * np.ones(Tn)))
for title, b_cr, oracle in [("best estimate at break-even (b = 100)", np.full(Tn, 100.0), (Z * sd_st[s] + np.maximum(mu_st[s] - 100, 0)) * p / 100),
                            ("best estimate state-conditioned (b = mu(r))", mu_st[s], Z * sd_st[s] * p / 100)]:
    print(f"\n{title}\n  {'rule':40s}{'all':>8s}{'soft':>8s}{'hard':>8s}{'avg cap':>9s}")
    for lab, ch in [("standard formula, premium volume", 3 * sigma_hat * p), ("exposure-corrected volume V*", 3 * sigma_hat * pbar * np.ones(Tn)),
                    ("state-conditional requirement", oracle)]:
        r = cov(ch, b_cr); print(f"  {lab:40s}{r[0]:7.3f}%{r[1]:7.3f}%{r[2]:7.3f}%{r[3]:9.2f}")
print(f"  ({time.time()-T0:.0f}s)")


# --------------------------------------------------------------------------
# Calibration of the loading (Sec. 6.2, Table 7): closed form in the two-state model
# Criterion of Sec. 6.1: equal exceedance probability in both states at unchanged
# average capital (panel A). Exact 0.5% coverage, which changes the level, is
# reported only as a diagnostic of the level of the factor (panel B, memo).
# --------------------------------------------------------------------------
banner("CAPITAL-NEUTRAL CALIBRATION OF THE LOADING AND LEVEL DIAGNOSTIC (Table 7)")
c_ind = np.log(pbar / p_state)                          # indicator by state (soft, hard); centred by construction
print(f"indicator: c_S {c_ind[0]:+.4f}  c_H {c_ind[1]:+.4f}  (ergodic mean {pi@c_ind:+.1e})")
def fmt_e(e):
    return "<0.001%" if e < 1e-5 else f"{100*e:.3f}%"
for dgp, s_r in [("A: price leverage only", np.array([sC, sC])),
                 ("B: soft-state dispersion at its estimated value", np.array([ms["sd"][0] * pS / 100, sC]))]:
    mu_r, sd_r = 100 * mH / p_state, 100 * s_r / p_state
    sig_h = np.sqrt(pi @ (sd_r ** 2) + pi[0] * pi[1] * (mu_r[0] - mu_r[1]) ** 2) / 100
    K0 = 300 * sig_h                                    # static charge per 100 of premium
    print(f"\nDGP {dgp}: sigma(S) {sd_r[0]:.2f}, sigma(H) {sd_r[1]:.2f}; pooled sigma_hat {sig_h:.4f}; static charge {K0:.2f} per 100 of premium")
    for basis, b in [("break-even (b = 100)", np.array([100.0, 100.0])), ("state-conditioned (b = mu(r))", mu_r)]:
        d = b * p_state / 100 - mH                      # provision less expected claims, per unit of exposure
        exc = lambda K: 1 - stats.norm.cdf((K * p_state / 100 + d) / s_r)
        avg = lambda K: pi @ (K * p_state / 100)        # average capital per unit of exposure (points of hard-state premium)
        A0 = avg(np.full(2, K0)); e0 = exc(np.full(2, K0))
        # panel A: equal exceedance in both states, average capital unchanged
        M = np.array([[1 / s_r[0], -1 / s_r[1]], [pi[0], pi[1]]]); rhs = np.array([d[1] / s_r[1] - d[0] / s_r[0], A0])
        X = np.linalg.solve(M, rhs); K = 100 * X / p_state
        R = K[0] / K[1]; gam = (R - 1) / (c_ind[0] - R * c_ind[1]); lam = K[1] / (K0 * (1 + gam * c_ind[1]))
        e1 = exc(lam * K0 * (1 + gam * c_ind))
        # smallest capital-neutral loading that meets the soft-state target
        def soft_gap(g):
            l = A0 / avg(K0 * (1 + g * c_ind)); return exc(l * K0 * (1 + g * c_ind))[0] - 0.005
        gmin = optimize.brentq(soft_gap, 0, 50) if e0[0] > 0.005 else 0.0
        lmin = A0 / avg(K0 * (1 + gmin * c_ind))
        # panel B (memo): exact 0.5% coverage in each state, level free
        target = (mH + Z * s_r - b * p_state / 100) / p_state * 100
        Aq = target / K0; lg = (Aq[0] - Aq[1]) / (c_ind[0] - c_ind[1]); lam_x = Aq[0] - lg * c_ind[0]; gam_x = lg / lam_x
        print(f"  {basis}:")
        print(f"    static rule                charge S {K0:5.1f} H {K0:5.1f}  exceedance S {fmt_e(e0[0])} H {fmt_e(e0[1])}  avg capital {A0:.2f}")
        print(f"    capital-neutral, equal     gamma {gam:.2f} lambda {lam:.3f}  charge S {lam*K0*(1+gam*c_ind[0]):5.1f} H {lam*K0*(1+gam*c_ind[1]):5.1f}  "
              f"exceedance S {fmt_e(e1[0])} H {fmt_e(e1[1])} (z = {stats.norm.isf(e1[0]):.2f})  avg capital {avg(lam*K0*(1+gam*c_ind)):.2f}")
        print(f"    smallest capital-neutral loading meeting the soft-state target: gamma {gmin:.2f} (lambda {lmin:.3f})")
        print(f"    memo, exact 0.5% coverage: lambda {lam_x:.3f} gamma {gam_x:.2f}  targets S {target[0]:.2f} H {target[1]:.2f}  "
              f"avg capital {avg(target):.2f} ({100*(avg(target)/A0-1):+.0f}% against static)")

# --------------------------------------------------------------------------
# Real-time checks (Sec. 6.3, Table 8)
# Panel A: the price-index rule (8) applied to a published index. The index is
# the BLS producer price index for direct property and casualty insurers
# (annual average of the monthly index, June 1998 = 100), deflated by the
# CPI-U (annual average). Indicator at the end of year t: log of the trailing
# k-year geometric mean (years t-k..t-1) less log of the year-t value, capped
# at +/- CBAR; it scales the volatility component of the charge for year t+1.
# Panel B: the regime model with filtered probabilities.
# --------------------------------------------------------------------------
banner("REAL-TIME CHECK A: PRICE-INDEX RULE (8), BLS PRODUCER PRICE INDEX (Table 8, panel A)")
K_WIN, CBAR = 7, 0.15
pidx = df.dropna(subset=["ppi_pc", "cpi_u"])
real_p = dict(zip(pidx.year.astype(int), pidx.ppi_pc.astype(float) / pidx.cpi_u.astype(float)))
rp0 = real_p[min(real_p)]
cr_of = dict(zip(yrs, y))
def indicator(t, k=K_WIN):
    if t - k not in real_p or t not in real_p:
        return None
    ref = np.mean([np.log(real_p[s_]) for s_ in range(t - k, t)])
    return float(np.clip(ref - np.log(real_p[t]), -CBAR, CBAR))
print("real price index (first year = 100): " + " ".join(f"{t}:{100*real_p[t]/rp0:.1f}" for t in sorted(real_p)))
ind = {t: indicator(t) for t in sorted(real_p) if indicator(t) is not None and t + 1 in cr_of}
print(f"indicator (k = {K_WIN}, cap {CBAR}): " + " ".join(f"{t}:{v:+.3f}" for t, v in ind.items()))
tt = np.array(sorted(ind)); cc = np.array([ind[t] for t in tt]); nxt = np.array([cr_of[t + 1] for t in tt])
print(f"charges for {tt[0]+1}-{tt[-1]+1} (n = {len(tt)}); indicator positive in {int((cc>0).sum())} years; mean indicator {cc.mean():+.4f}; "
      f"max {cc.max():+.3f} ({tt[cc.argmax()]}), min {cc.min():+.3f} ({tt[cc.argmin()]})")
print(f"corr(indicator_t, CR_t+1) {np.corrcoef(cc, nxt)[0,1]:.2f}; mean CR_t+1 when indicator > 0: {nxt[cc>0].mean():.2f} "
      f"(n = {(cc>0).sum()}), otherwise {nxt[cc<=0].mean():.2f} (n = {(cc<=0).sum()}); "
      f"CR > 100 in {int((nxt[cc>0]>100).sum())} of {(cc>0).sum()} and {int((nxt[cc<=0]>100).sum())} of {(cc<=0).sum()} years")
rows_a = []
for t, cv in zip(tt, cc):
    hist = y[yrs <= t]; m_, s_ = hist.mean(), hist.std(ddof=1)
    for pr in (0.90, 0.95, 0.995):
        zq = stats.norm.ppf(pr)
        for gam_ in (0.0, 1.0, 4.0):
            ch = (m_ - 100) + zq * s_ * (1 + gam_ * cv)
            rows_a.append(dict(year=t + 1, p=pr, gamma=gam_, indicator=cv, realised=cr_of[t + 1], charge=ch, exc=int(cr_of[t + 1] - 100 > ch)))
pa = pd.DataFrame(rows_a); pa.to_csv(os.path.join(OUT, "realtime_price_index.csv"), index=False)
print(f"{'rule':30s}{'exc 90%':>9s}{'exc 95%':>9s}{'exc 99.5%':>11s}{'mean 99.5%':>12s}{'range 99.5%':>14s}{'corr(charge, CR-100)':>22s}")
for gam_, lab in [(0.0, "static (gamma = 0)"), (1.0, "adjusted, gamma = 1"), (4.0, "adjusted, gamma = 4")]:
    d = pa[pa.gamma == gam_]; d995 = d[d.p == 0.995]
    print(f"{lab:30s}{d[d.p==0.90].exc.sum():9d}{d[d.p==0.95].exc.sum():9d}{d995.exc.sum():11d}{d995.charge.mean():12.2f}"
          f"{d995.charge.min():8.1f}-{d995.charge.max():4.1f}{np.corrcoef(d995.charge, d995.realised - 100)[0,1]:22.2f}")
st995 = pa[(pa.p == 0.995) & (pa.gamma == 0)].charge.values
for gam_ in (1.0, 4.0):
    a995 = pa[(pa.p == 0.995) & (pa.gamma == gam_)].charge.values
    dlt = a995 - st995
    print(f"gamma = {gam_:.0f}: mean 99.5% charge {100*(a995.mean()/st995.mean()-1):+.1f}% against static; largest increase {dlt.max():.1f} points "
          f"({int(tt[np.argmax(dlt)])+1}); largest decrease {dlt.min():.1f} points ({int(tt[np.argmin(dlt)])+1})")
print(f"expected exceedances over {len(tt)} years: 90% {0.10*len(tt):.2f}, 95% {0.05*len(tt):.2f}, 99.5% {0.005*len(tt):.2f}")
for kk in (5, 10):
    ii = {t: indicator(t, kk) for t in sorted(real_p) if indicator(t, kk) is not None and t + 1 in cr_of}
    t2 = np.array(sorted(ii)); c2 = np.array([ii[t] for t in t2]); n2 = np.array([cr_of[t + 1] for t in t2])
    print(f"window k = {kk}: charges for {t2[0]+1}-{t2[-1]+1}, corr(indicator_t, CR_t+1) {np.corrcoef(c2, n2)[0,1]:.2f}, "
          f"mean CR_t+1 {n2[c2>0].mean():.2f} when positive vs {n2[c2<=0].mean():.2f}, mean indicator {c2.mean():+.4f}")
# Fig. S1 (supplementary)
fig, ax = plt.subplots(2, 1, figsize=(6.5, 4.6), dpi=300, sharex=True, gridspec_kw={"height_ratios": [1, 1]})
ax[0].plot(sorted(real_p), [100 * real_p[t] / rp0 for t in sorted(real_p)], color="black", lw=1.4, marker="o", ms=2.4)
ax[0].set_ylabel(f"Real price index\n({min(real_p)} = 100)"); ax[0].spines[["top", "right"]].set_visible(False)
ax0b = ax[0].twinx(); ax0b.bar(tt, cc, color="0.75", width=0.6); ax0b.axhline(0, color="0.4", lw=0.6)
ax0b.set_ylabel("Indicator $c_t$ (bars)"); ax0b.spines[["top"]].set_visible(False); ax0b.set_ylim(-0.12, 0.12)
ax[0].set_zorder(ax0b.get_zorder() + 1); ax[0].patch.set_visible(False)
ax[1].plot(yrs[yrs >= 1999], y[yrs >= 1999], color="black", lw=1.4, marker="o", ms=2.4)
ax[1].axhline(100, color="0.4", lw=0.8, ls="--"); ax[1].set_ylabel("Combined ratio (%)"); ax[1].set_xlabel("Year")
ax[1].spines[["top", "right"]].set_visible(False); ax[1].set_xlim(1998.4, 2025.8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "figS1_price_index.png")); plt.close()
print("figure written: figS1_price_index.png")

banner("REAL-TIME CHECK B: REGIME MODEL, FILTERED PROBABILITIES, EXPANDING WINDOW (Table 8, panel B)")
def predictive_q(hist, pr):
    r = em(hist, K=2, switching_var=True, n_starts=8, seed=7, do_polish=False)
    w = r["filtered"][-1] @ r["P"]
    grid_q = np.linspace(60, 180, 24001)
    cdf = (w[None, :] * stats.norm.cdf(grid_q[:, None], r["mu"][None, :], r["sd"][None, :])).sum(1)
    return grid_q[np.searchsorted(cdf, pr)], w[0]
rows = []
for t in range(15, len(y)):
    hist, real = y[:t], y[t]
    for pr in (0.90, 0.95, 0.995):
        qa, wsoft = predictive_q(hist, pr); qs = hist.mean() + stats.norm.ppf(pr) * hist.std(ddof=1)
        rows.append(dict(year=yrs[t], p=pr, realised=real, w_soft=wsoft, adj=qa - 100, static=qs - 100, adj_exc=int(real > qa), static_exc=int(real > qs)))
bt = pd.DataFrame(rows); bt.to_csv(os.path.join(OUT, "backtest.csv"), index=False)
for pr in (0.90, 0.95, 0.995):
    d = bt[bt.p == pr]
    print(f"p = {pr:.3f}: n = {len(d)}, expected exceedances {(1-pr)*len(d):.2f}; adjusted {d.adj_exc.sum()} (mean charge {d.adj.mean():.2f}), static {d.static_exc.sum()} (mean charge {d.static.mean():.2f})")
print(f"year-by-year at 99.5%: " + "; ".join(f"{int(r.year)} adj {r.adj:.1f} static {r.static:.1f}" for _, r in bt[bt.p == 0.995].iterrows()))
d995 = bt[bt.p == 0.995]
print(f"corr(charge, CR-100) at 99.5%: adjusted {np.corrcoef(d995.adj, d995.realised-100)[0,1]:.2f}, static {np.corrcoef(d995.static, d995.realised-100)[0,1]:.2f}")
print(f"\nfull run complete ({time.time()-T0:.0f}s)")
