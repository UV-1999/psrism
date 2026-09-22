# PSRISM

`psrism` is a command-line Python package for pulsar interstellar-medium analysis from PSRCHIVE-readable archive files. It is designed for studying pulse broadening, diffractive scintillation, scintillation bandwidths and timescales, secondary spectra, and the frequency scaling of scattering in radio pulsar observations.

Scientifically, `psrism` turns a calibrated pulsar archive into analysis products such as measured dispersion measures, dynamic spectra, autocorrelation spectra, secondary spectra, integrated pulse profiles, scattering timescales τ, scattering spectral indices α, intrinsic Gaussian-width evolution, and scattering-aware residual-DM checks. It uses `psrchive` for archive I/O and standard scientific Python tools for numerical work, fitting, and plotting. Python dependencies are listed in [`requirements.txt`](requirements.txt). `psrchive` and its `pdmp` program are usually installed separately, for example through conda-forge or a local PSRCHIVE build.

## Scientific conventions

PSRISM uses the following literature-facing conventions throughout calculations, terminal output, plots, and batch tables:

| Quantity | Convention and unit |
| --- | --- |
| Observing frequency | MHz |
| Scattering time `tau` | seconds |
| Scattering index `alpha` | `tau(nu) = tau_ref (nu / nu_ref)^(-alpha)`; ordinary scattering has positive `alpha` [1, §4.2.3] |
| Reference scattering time | `tau150` by default: `tau_ref` at `nu_ref = 150 MHz`; change with `--tau-reference-freq` [3, PDF pp. 34-35, §2.7.4] |
| Intrinsic profile width | FWHM of each fitted intrinsic Gaussian, reported in bins, phase, seconds, and percentage of pulse period [2, PDF pp. 11-12, Appendix B; PDF p. 12, Fig. B1] |
| ACF decorrelation bandwidth | Positive frequency-lag half-width at `ACF = 1/2`, in MHz [1, §7.4.4.1] |
| ACF diffractive timescale | Positive time-lag half-width at `ACF = 1/e`, in seconds [1, §7.4.4.1] |
| ACF drift slope | `d(time lag) / d(frequency lag)`, in s/MHz; positive means time lag increases with frequency lag |
| Secondary-spectrum axes | Fringe frequency in Hz and delay in seconds |
| Arc curvature `eta` | `delay - delay0 = eta (fringe frequency - fringe0)^2`, in s^3 |
| Dispersion measure | pc cm^-3 |
| Distance and transverse velocity | kpc and km/s |

For arc fitting, `--arc-half positive` searches positive delay and returns positive curvature; `negative` searches negative delay and returns negative curvature; `both` searches both halves and reports the positive curvature magnitude. These constants are defined in [`psrism/conventions.py`](psrism/conventions.py).

### Citation and provenance policy

Scientific definitions, equations, conventions, hardcoded scientific values, and literature-selected methods are followed by numbered citations. A Handbook citation gives the lowest applicable section number; a citation to another bundled PDF gives the PDF page, and an equation or section number where available. The numbered bibliography is in [References](#references).

Values described as a **PSRISM choice** are implementation decisions rather than claims taken from the cited literature. Examples include numerical optimizer bounds, robust-RFI thresholds, display floors, and arc-search grid settings. Matching `Reference:` and `PSRISM choice:` comments are kept beside the relevant source code. Ordinary figure sizes, colors, line widths, and output resolution are presentation settings and are not scientific constants.

### Compatibility with earlier PSRISM results

Earlier local revisions used `tau(nu) = tau0 (nu / nu0)^alpha`, so a scattering law that decreased with frequency produced a negative fitted index. The following sign conversion is PSRISM compatibility algebra:

```math
\alpha_{\rm current}=-\alpha_{\rm old}.
```

The default reference frequency has also changed. Earlier archive workflows used the median fitted subband frequency, while direct calls to `fit_alpha` defaulted to 1000 MHz. Current workflows consistently use 150 MHz unless `--tau-reference-freq` is supplied. The following conversion is algebraically derived from the cited power law. [1, §4.2.3]

```math
\tau_{150}
=
\tau_{\rm old,ref}
\left(
\frac{150\ {\rm MHz}}{\nu_{\rm old,ref}}
\right)^{-\alpha_{\rm current}}.
```

Old terminal logs normally contain `nu0`, which supplies `nu_old,ref` for this conversion. If that frequency is unavailable, rerunning the fit is safer than treating the old `tau0` as `tau150`. Old and current alpha columns must not be combined without reversing the old signs.

The implementation is enforced in three places: shared constants define the 150 MHz and ACF conventions, the log-space design matrix fits `-alpha` while returning positive `alpha`, and regression tests check the alpha, ACF, drift, and arc-sign behavior. The command line prints the equation with every alpha result, and directory CSV output records `tau_reference_frequency_mhz` beside `tau0_s`.

## install

Clone the repository, enter the package directory, and install the CLI in editable mode:

```bash
git clone https://github.com/UV-1999/psrism.git
cd psrism
python -m pip install -e .
```

Check that the command is available:

```bash
psrism --help
```

If `pip` is not available in the active conda environment:

```bash
conda install -c conda-forge pip
python -m pip install -e .
```

You can also run the package as a module from the repository directory:

```bash
python -m psrism --help
```

## Features

### Output Organization

Every run creates or reuses a folder named after the pulsar in the current working directory. `psrism` uses the PSRCHIVE source name when available, and falls back to the archive filename prefix otherwise.

All generated plots and the terminal-output log are written into that folder:

```text
PULSAR_NAME/
  ARCHIVE_STEM_terminal_output.txt
  ARCHIVE_STEM_dm_measurement.json
  ARCHIVE_STEM_data_quality.json
  ARCHIVE_STEM_scintillation.json
  ARCHIVE_STEM_arc_fit.json
  ARCHIVE_STEM_intrinsic_width.json
  ARCHIVE_STEM_anisotropic_intrinsic_width.json
  PULSAR_time_series_variability.json
  PULSAR_time_series_dm_solar.json
  PULSAR_time_series_annual_anisotropy.json
  PULSAR_time_series_annual_anisotropy_chain.npz
  ARCHIVE_ROOT_psrism_batch/ARCHIVE_ROOT_batch_manifest.json
  ARCHIVE_ROOT_psrism_batch/ARCHIVE_ROOT_batch_observations.csv
  ARCHIVE_ROOT_psrism_batch/ARCHIVE_ROOT_batch_pulsars.csv
  ARCHIVE_ROOT_psrism_batch/ARCHIVE_ROOT_batch_failures.csv
  ARCHIVE_ROOT_psrism_batch/ARCHIVE_ROOT_batch_alpha_vs_tau_ref_fraction.png
  ARCHIVE_STEM_dspec.png
  ARCHIVE_STEM_acspec.png
  ARCHIVE_STEM_sspec.png
  ARCHIVE_STEM_arc_search.png
  ARCHIVE_STEM_intrinsic_width_vs_freq.png
  ARCHIVE_STEM_anisotropic_intrinsic_width_vs_freq.png
  PULSAR_time_series_PARAMETER_autocorrelation.png
  PULSAR_time_series_correlations.png
  PULSAR_time_series_dm_vs_sun_separation.png
  PULSAR_time_series_dm_piecewise_slopes.png
  PULSAR_time_series_annual_anisotropy.png
  PULSAR_time_series_annual_anisotropy_posterior.png
  ...
```

The terminal output is still shown on screen, but the same text is also saved in `*_terminal_output.txt`. Screen output uses ANSI colors for section headings, warnings, errors, and important paths. The saved text log strips those ANSI codes so it remains easy to read and parse. Set `PSRISM_NO_COLOR=1` to disable colored terminal output. If the command is run again on the same archive, same-named files in the pulsar folder are overwritten there, keeping the working directory itself clean.

### Archive Only

The minimal command is:

```bash
psrism ARCHIVE
```

With only an archive specified, `psrism`:

- loads the archive with `psrchive.Archive_load`;
- records the original archive dimensions;
- applies the default preprocessing used by the package: dedispersion and polarization scrunching;
- applies optional `--dm`, `--nsub`, `--nchan`, and `--nbin` choices if supplied;
- prints original dimensions, processed dimensions, observing time, center frequency, bandwidth, frequency range, dispersion measure, and telescope name;
- does not make plots unless a plotting or fitting flag is requested.

This behavior is implemented in:

```text
psrism/cli.py
psrism/archive_io.py
```

### Inspect Mode

Before choosing scrunching values, use:

```bash
psrism ARCHIVE --inspect
```

This mode loads only archive metadata and prints valid smaller scrunch targets for `--nsub`, `--nchan`, and `--nbin`. PSRCHIVE scrunch targets must divide the current dimension exactly. For example, if an archive has `nchan=366`, valid smaller `--nchan` targets include:

```text
1, 2, 3, 6, 61, 122, 183
```

The divisor logic is implemented in:

```text
psrism/archive_io.py
```

using `valid_scrunch_targets`, `suggested_scrunch_targets`, and `format_scrunch_targets`.

You can also inspect an archive with PSRCHIVE directly:

```bash
vap -c file,nsub,npol,nchan,nbin,dm,freq,bw ARCHIVE
```

### Directory Time-Series Mode

If the positional input is a directory instead of a single archive, `psrism` treats the directory as a chronological set of archives for one pulsar and makes parameter-versus-time products. The default file pattern is `*.nop`.

Run all available time-series parameters:

```bash
psrism /home/piyushmarmat/PhD/old/Data/J0139+5814 \
  --time-params all \
  --nsub 70 \
  --nchan 61 \
  --nbin 256 \
  --tau-subbands 4
```

Run a lighter time series with only archive metadata and profile-fitting parameters:

```bash
psrism /home/piyushmarmat/PhD/old/Data/J0139+5814 \
  --time-params dm,tau,alpha
```

Restrict the run to one archive or one subset of archives:

```bash
psrism /home/piyushmarmat/PhD/old/Data/J0139+5814 \
  --time-pattern "J0139+5814_2019-08*.nop" \
  --time-params dnu_d,dt_d,t_r \
  --nsub 70 \
  --nchan 61 \
  --nbin 256
```

The supported parameter names are:

```text
dm       measured pdmp DM with --measure-dm; otherwise archive metadata DM
tau      full-band integrated-profile scattering timescale
tau_ref  scattering timescale from the subband power-law fit at --tau-reference-freq
alpha    scattering spectral index from subband tau fits
dnu_d    ACF decorrelation bandwidth
dt_d     ACF diffractive scintillation timescale
t_r      refractive scintillation timescale inferred from ACF dnu_d and dt_d
all      all of the above
```

`tau_ref` may also be requested as `tau150` or `tau0`. Its reference frequency defaults to 150 MHz, so the resulting time series is the thesis quantity `tau150`; changing `--tau-reference-freq` changes both its value and plotted interpretation. [3, PDF pp. 34-35, §2.7.4] It is distinct from `tau`, which comes from one fit to the frequency-integrated profile. Selecting either `tau_ref` or `alpha` runs the same quality-controlled subband power-law fit, and `all` includes both scattering-time measurements.

To replace metadata DMs with per-epoch measurements and show their reported uncertainties as plot error bars:

```bash
psrism ARCHIVE_DIR --time-params dm --measure-dm
```

The measured value is applied to the in-memory archive before any requested tau, alpha, or scintillation analysis. The source archive file is never overwritten. `pdmp` integration, its setup, controls, and the additional CSV columns are documented in [Dispersion-Measure Refinement](#dispersion-measure-refinement).

Epochs are read from the PSRCHIVE integration epoch when available. If that fails, `psrism` falls back to the archive filename pattern `YYYY-MM-DD_HH:MM:SS`. The MJD epoch below is an astronomical time-standard definition, not a value taken from the bundled pulsar references:

```math
{\rm MJD}
=
\frac{
t_{\rm UTC}
-
{\rm 1858\text{-}11\text{-}17\ 00:00:00\ UTC}
}{
86400\ {\rm s}
}.
```

Each time-series plot shows UTC calendar dates on the bottom x-axis and MJD on the top x-axis. The plot title includes the pulsar/directory name and the observed band edges in MHz, for example:

```text
J0139+5814 (118.066-189.551 MHz): tau vs time
```

`--time-axis utc` and `--time-axis mjd` are retained as legacy-compatible inputs. Current plots always show both scales, so this option no longer changes the output.

The time-series records include the Sun-pulsar angular separation because solar angular distance is a possible covariate of scattering and DM variability. [3, PDF p. 35, §2.8] PSRISM calculates the geocentric Sun position with Astropy's built-in ephemeris, matching the software choice described for the reference DM analysis. [3, PDF p. 70, §4.7] It stores start-, midpoint-, and end-of-observation separations and `solar_ephemeris_method` in the CSV. If Astropy is unavailable in an older environment, PSRISM retains its low-precision geocentric approximation as an explicitly labelled `psrism_approximate_fallback`; this fallback and the approximate station table are PSRISM utility choices and are not taken from the bundled references.

For each archive, the directory workflow saves one CSV table and one plot for each requested parameter:

```text
J0139+5814/
  J0139+5814_time_series_terminal_output.txt
  J0139+5814_time_series.csv
  J0139+5814_time_series_dm_vs_time.png
  J0139+5814_time_series_tau_vs_time.png
  J0139+5814_time_series_variability.json
  J0139+5814_time_series_dm_solar.json
  J0139+5814_time_series_tau_autocorrelation.png
  J0139+5814_time_series_correlations.png
  J0139+5814_time_series_dm_vs_sun_separation.png
  J0139+5814_time_series_dm_piecewise_slopes.png
  J0139+5814_time_series_annual_anisotropy.json
  J0139+5814_time_series_annual_anisotropy_chain.npz
  J0139+5814_time_series_annual_anisotropy.png
  J0139+5814_time_series_annual_anisotropy_posterior.png
  ...
```

The ACF-derived decorrelation bandwidth and diffractive timescale use the same definitions as the single-archive `--acspec` path:

```math
\Delta\nu_{\rm d}:
{\rm ACF}(\Delta\nu,0)=\frac{1}{2},
```

```math
\Delta t_{\rm d}:
{\rm ACF}(0,\Delta t)=\frac{1}{e}.
```

These half-width definitions are the standard conventions in the Handbook. [1, §7.4.4.1]

Directory mode uses the same tilted 2D Gaussian measurement and resolution checks as `--acspec`, rather than treating the one-dimensional slice interpolation as the primary result. The accepted `dnu_d` and `dt_d` values have covariance errors in `decorrelation_bandwidth_error_mhz` and `diffractive_timescale_error_s`, which are shown as time-series error bars. The CSV retains the raw fitted values, slice crossings, drift slope, fit correlation, sample resolutions, width-to-resolution ratios, method, and resolved flags even when a value is rejected. Full field details are given in [Autocorrelation Spectrum](#autocorrelation-spectrum).

The refractive timescale plotted as `t_r` is computed from those measured ACF quantities using the quoted refractive/diffractive relation: [5, PDF p. 9, Eq. 14]

```math
T_{\rm r}
=
\frac{4}{\pi}
\frac{\nu}{\Delta\nu_{\rm d}}
\Delta t_{\rm d}.
```

Here `ν` and `Δν_d` are both in MHz, so `T_r` has the same time unit as `Δt_d`; the plot reports it in days. PSRISM propagates the joint 2D-fit covariance of `Δν_d` and `Δt_d` through this ratio, stores the result as `refractive_timescale_error_days`, and shows it on the time-series plot. First-order covariance propagation is a PSRISM choice.

Implementation:

```text
psrism/cli.py
psrism/time_series_analysis.py
psrism/autocorrelation_spectrum.py
psrism/dynamic_spectrum.py
psrism/fit_tau.py
psrism/fit_alpha.py
psrism/dm_variability.py
```

### Cross-Pulsar Batch Mode

Ordinary directory mode assumes that all matching archives belong to one pulsar. A common server path can instead contain files from many pulsars, possibly in nested directories. Use explicit batch mode for that layout so measurements from different sources are never combined into one time series:

```bash
psrism /QNAP/LOFAR/PL611 \
  --batch-pulsars \
  --batch-recursive \
  --time-pattern "*.nop" \
  --time-params dm,tau,tau_ref,alpha
```

`--batch-pulsars` first opens each matching archive and groups it by a `B...` or `J...` pulsar name in the PSRCHIVE header. A leading `PSR` label is normalized away. If the header source is empty, the pulsar name may be inferred from the filename. Readable calibrator and other non-pulsar archives are counted and ignored; files with no usable source information are recorded as source-discovery failures. `--batch-recursive` changes discovery from top-level `glob` to recursive `rglob`. These grouping, normalization, and fallback rules are PSRISM operational choices, not scientific methods from the references.

Restrict an expensive run to selected pulsars by repeating `--batch-source`:

```bash
psrism /QNAP/LOFAR/PL611 \
  --batch-pulsars --batch-recursive \
  --batch-source J0139+5814 \
  --batch-source J0614+2229 \
  --time-pattern "*.nop" \
  --time-params all
```

Add `--fit-anisotropy` when both isotropic and anisotropic subband models should be fitted for every usable epoch:

```bash
psrism /QNAP/LOFAR/PL611 \
  --batch-pulsars --batch-recursive \
  --batch-source J0139+5814 \
  --time-pattern "*.nop" \
  --time-params alpha,tau_ref \
  --fit-anisotropy
```

In directory and batch modes this flag adds anisotropic `alpha`, `tau_ref`, uncertainty, reference-frequency, and accepted/rejected/failed subband columns to the per-epoch CSV. The ordinary `alpha` and `tau_ref` columns remain the isotropic results, preventing model provenance from being lost.

Source matching is case-insensitive. Every selected pulsar is sent independently through the existing directory pipeline, so its epoch ordering, quality checks, fitted quantities, temporal statistics, plots, and citations remain exactly those documented in the preceding sections. A failure in one pulsar group does not stop later groups.

For an input directory named `PL611`, batch products are organized as:

```text
PL611_psrism_batch/
  PL611_batch_terminal_output.txt
  PL611_batch_manifest.json
  PL611_batch_observations.csv
  PL611_batch_pulsars.csv
  PL611_batch_failures.csv
  PL611_batch_alpha_vs_tau_ref_fraction.png
  J0139+5814/
    J0139+5814_time_series.csv
    J0139+5814_time_series_terminal_output.txt
    J0139+5814_time_series_*.json
    J0139+5814_time_series_*.png
  J0614+2229/
    J0614+2229_time_series.csv
    J0614+2229_time_series_terminal_output.txt
    ...
```

The top-level exports form the PSRISM-to-PSRDEX handoff:

- `*_batch_observations.csv` combines every successfully processed epoch and adds a leading `pulsar` column while retaining all per-epoch provenance and quality fields.
- `*_batch_pulsars.csv` contains one row per pulsar with discovered, processed, and skipped counts; MJD, frequency, and pulse-period coverage; medians of available quantities; and weighted isotropic and optional anisotropic `alpha` and `tau_ref` summaries.
- `*_batch_failures.csv` records source-discovery failures and archives skipped during per-pulsar processing.
- `*_batch_manifest.json` is strict JSON containing the discovery settings, selected sources, counts, run status, archive lists, and paths to each per-pulsar table and log.
- `*_batch_alpha_vs_tau_ref_fraction.png` compares weighted `alpha` with weighted `tau_ref / P` for every pulsar and available scattering model.

The reference work combines the epoch measurements for each pulsar using the inverse square of their uncertainties and compares weighted `alpha` against weighted `tau150` as a fraction of pulse period. [3, PDF pp. 43-44, §3.2, Table 3.1 and Fig. 3.1] PSRISM applies the same weighted-mean calculation separately to the isotropic and anisotropic columns:

```math
\bar{x}_w
=
\frac{\sum_i x_i/\sigma_i^2}{\sum_i 1/\sigma_i^2},
\qquad
\sigma_{\bar{x}_w}
=
\left(\sum_i\frac{1}{\sigma_i^2}\right)^{-1/2}.
```

The per-source pulse period is retained from each archive and its median is used as `P`; using the median period and carrying the weighted-mean error into `tau_ref / P` are PSRISM catalogue choices. The additional summary medians are compact, outlier-resistant values for browsing and filtering. They are not fitted population relations and must not replace the per-epoch measurements in physical inference. The batch layer deliberately performs no cross-pulsar regression because observing coverage, frequency, uncertainty, and selection effects differ between sources.

Batch mode returns exit status `0` when every discovered archive and pulsar group succeeds, `1` when usable products were exported with recorded partial failures, and `2` when no pulsar group succeeds. The CSV and manifest products are still written on partial success.

The following source-specific options are rejected in mixed-source mode:

- `--dm VALUE`, because one manual DM must not be applied to multiple pulsars; use archive metadata or `--measure-dm`;
- `--intrinsic-template`, because a pulse template belongs to one source;
- `--fit-annual-anisotropy`, because distance and proper motion must be supplied independently for each pulsar.

Implementation:

```text
psrism/batch_analysis.py
psrism/cli.py
psrism/time_series_analysis.py
```

#### Temporal Variability Statistics

Every directory run now applies the same non-ML temporal analysis to each parameter selected by `--time-params`. The thesis analyzes whether the measured `tau150` and `alpha` series contain structured variation rather than only white noise, using autocorrelation, the runs test, and the Ljung-Box test. [3, PDF pp. 51-55, §§4.1-4.2] PSRISM exposes that same reference-frequency series as `tau_ref`, rather than substituting the distinct full-band `tau` measurement. The thesis also compares temporal `DM`, `tau150`, and `alpha` with Pearson coefficients and their p-values. [3, PDF pp. 73-74, §4.8]

PSRISM does not add ADF or KPSS stationarity tests to this reference-reproduction workflow: the cited white-noise analysis uses ACF, runs, and Ljung-Box tests, while ADF and KPSS test different stationarity or unit-root null hypotheses. They can be added later as explicitly separate PSRISM diagnostics if a concrete scientific use requires them.

For each parameter, PSRISM reports the number of usable epochs, MJD range and span, arithmetic mean, sample standard deviation, median, minimum, maximum, and an inverse-variance weighted mean when positive finite uncertainties exist. The reference analysis compares temporal measurements with their weighted mean. [3, PDF p. 55, §4.2] PSRISM specifically uses:

```math
\bar{x}_{w}
=
\frac{\sum_i x_i/\sigma_i^2}{\sum_i 1/\sigma_i^2},
\qquad
\sigma_{\bar{x}_{w}}
=
\left(\sum_i\frac{1}{\sigma_i^2}\right)^{-1/2}.
```

The inverse-variance definition is a PSRISM implementation choice. The parameter-versus-time plot displays this weighted mean as a dashed line; if fewer than two usable uncertainties exist, it displays the arithmetic mean instead.

PSRISM also tests the uncertainty-weighted constant model:

```math
\chi^2_{\rm constant}
=
\sum_i
\left(
\frac{x_i-\bar{x}_{w}}{\sigma_i}
\right)^2,
\qquad
\mathrm{dof}=N_{\sigma}-1.
```

The chi-square survival probability is reported as `constant_p_value`, and `constant_rejected` is true when it falls below `--variability-p-threshold`. This constant-model diagnostic is a PSRISM addition, not a test specified in the bundled references.

PSRISM evaluates the discrete sample autocorrelation in chronological observation order:

```math
\rho_k
=
\frac{
\sum_{i=1}^{N-k}(x_i-\bar{x})(x_{i+k}-\bar{x})
}{
\sum_{i=1}^{N}(x_i-\bar{x})^2
}.
```

This is the sampled counterpart of the thesis autocorrelation definition. [3, PDF p. 52, §4.1.1, Eq. 4.1] The plotted confidence band uses the lag-dependent Bartlett approximation motivated in the same analysis: [3, PDF pp. 52-53, §4.1.1, Eqs. 4.2-4.3]

```math
C_k
=
z_{1-p/2}
\sqrt{
\frac{1+2\sum_{j=1}^{k-1}\rho_j^2}{N}
}.
```

These are observation-index lags, not elapsed-day lags. This matters for irregularly sampled monitoring: lag 1 means adjacent retained observations even if their time separation changes. The JSON records `lag_unit = observation_index`, and the ACF plot labels the axis accordingly.

The Wald-Wolfowitz runs diagnostic divides finite measurements above and below their median, removes ties, and calculates the expected run count and variance. The thesis gives the runs statistic and its null expectation and variance. [3, PDF pp. 53-54, §4.1.1, Eqs. 4.4-4.6] PSRISM reports a two-sided normal-approximation p-value. The thesis cautions that this test alone is not appropriate for separating deterministic variation from measurement noise, so PSRISM retains it as a diagnostic and never treats it as proof of physical variability. [3, PDF p. 54, §4.1.1]

For serial correlation, PSRISM calculates the Ljung-Box statistic at every retained lag: [3, PDF p. 54, §4.1.1, Eq. 4.7]

```math
Q_h
=
N(N+2)
\sum_{k=1}^{h}
\frac{\rho_k^2}{N-k}.
```

The JSON stores every `Q_h` and chi-square p-value, plus the minimum, maximum, and final-lag p-values. `ljung_box_independence_rejected` applies the threshold to the configured final lag, while `ljung_box_all_lags_rejected` reproduces the thesis table's stricter convention of requiring even the largest tested p-value to fall below the threshold. [3, PDF p. 55, §4.2, Table 4.3] By default PSRISM evaluates all possible nonzero lags through `N-1`; it stops before `N` because the quoted equation has a zero denominator there. An explicitly requested lag larger than `N-1` is capped at `N-1` for that parameter. Limit the analysis when only shorter correlations are meaningful:

```bash
psrism ARCHIVE_DIR --time-params tau,alpha --variability-max-lag 10
```

The default decision threshold is `p = 0.05`, matching the threshold used in the thesis variability table. [3, PDF p. 55, §4.2] It can be changed explicitly:

```bash
psrism ARCHIVE_DIR --variability-p-threshold 0.01
```

For every selected parameter pair with at least three pairwise-complete, nonconstant measurements, PSRISM reports the Pearson coefficient, two-sided p-value, paired sample count, and threshold flag. This follows the thesis comparison of `DM`, `tau150`, and `alpha`; the thesis also states the uncorrelated, normally distributed null assumptions and warns that most reported correlations were not statistically significant. [3, PDF pp. 73-74, §4.8] PSRISM extends the same calculation to any selected pair among `dm`, `tau`, `tau_ref`, `alpha`, `dnu_d`, `dt_d`, and `t_r`.

Missing or rejected epoch values are excluded per parameter, and correlations use pairwise-complete epochs. No multiple-comparison correction is applied, so pairwise p-values are exploratory diagnostics rather than standalone physical detections. These missing-data and interpretation rules are PSRISM choices.

The generated products are:

```text
*_time_series_variability.json
*_time_series_PARAMETER_autocorrelation.png
*_time_series_correlations.png
```

The JSON contains all descriptive values, test statistics, p-values, decisions, ACF values, confidence limits, Ljung-Box sequences, and pairwise Pearson results. Constant series retain an ACF value of one at lag zero and JSON `null` at undefined nonzero lags, avoiding non-standard `NaN` output. The correlation heatmap annotates each usable cell with `r`, `p`, and paired `n`.

Implementation:

```text
psrism/temporal_statistics.py
psrism/time_series_analysis.py
psrism/cli.py
```

#### DM Variability and Solar Separation

When `dm` is selected in a directory run, PSRISM adds a dedicated non-ML analysis of DM trends and their relation to the Sun. The reference work measures each epoch's DM by maximizing profile S/N with PSRCHIVE `pdmp`, then compares the resulting time series with Sun-pulsar angular separation. [3, PDF pp. 68-70, §§4.6-4.7] Use measured rather than header DMs with:

```bash
psrism ARCHIVE_DIR --time-params dm --measure-dm
```

The default `--dm-analysis-series dm` analyzes the same `dm` and `dm_error` columns used by the ordinary time-series plot. The alternative below analyzes `scattering_corrected_dm` from `--check-scattering-dm`; PSRISM reports a command error if the corrected series is selected without that required option:

```bash
psrism ARCHIVE_DIR --time-params dm --measure-dm --check-scattering-dm \
  --dm-analysis-series scattering-corrected
```

For the corrected series, PSRISM combines the `pdmp` DM error and residual scattering-DM error in quadrature when both are available, or retains the one available uncertainty. This propagation assumes independent errors and is a PSRISM choice. The scattering correction remains a diagnostic: it does not modify and reprocess the archives.

PSRISM reports the minimum Sun-pulsar separation, the number of epochs below `--solar-warning-angle`, and the Pearson coefficient, two-sided p-value, and paired sample count between DM and midpoint separation. The reference analysis uses Astropy for the angular separation and the Pearson coefficient and p-value for this comparison. [3, PDF p. 70, §4.7] Pearson values require at least three pairwise-complete epochs and variation in both quantities; otherwise the JSON fields are `null`. The significance flag uses `--variability-p-threshold`, so the solar and other temporal-correlation decisions share one declared threshold.

The default warning angle is 10 degrees because the reference discussion says solar-wind influence becomes stronger below approximately that separation. [3, PDF p. 85, §5.5.1] Change it without changing the calculated angles or removing any measurements:

```bash
psrism ARCHIVE_DIR --time-params dm --solar-warning-angle 15
```

This is a diagnostic threshold, not a data-selection cut. PSRISM does not subtract a solar-wind or ionospheric model. The thesis sample never approached within 40 degrees of the Sun, found no significant DM-separation correlation, and notes a maximum expected ionospheric contribution of approximately `5e-4 pc cm^-3`; those findings are properties of that sample, not universal correction constants. [3, PDF pp. 70, 85, §§4.7 and 5.5.1]

PSRISM also fits one global linear DM trend and automatically divides the chronological finite series into sections whose adjacent changes retain the same direction. The reference method manually divides a longer series into same-direction sections because one global slope can hide changes, computes an absolute DM derivative for each section, and combines them using the number of measurements in each section. [3, PDF pp. 71-72, §4.7.2] PSRISM's automatic segmentation is an explicit implementation choice:

```math
|DM_{i+1}-DM_i|
\leq
k\sqrt{\sigma_i^2+\sigma_{i+1}^2}
```

is treated as an uncertainty-compatible flat step, where `k` is `--dm-turning-sigma` and defaults to one. Flat steps attach to the preceding direction, or to the following direction at the beginning, and do not create a turn. If either adjacent uncertainty is unavailable, the sign of the measured difference is used directly. A direction reversal creates a breakpoint, and the turning-point measurement belongs to both neighboring fits so each fitted section includes the observed extremum. These rules are PSRISM choices.

Each section is fit as DM versus centered MJD, with slope reported in `pc cm^-3 yr^-1` using `365.25 days yr^-1`. A fit uses inverse-variance weighting only when every point in that section has a positive finite uncertainty; otherwise it uses all finite points in an unweighted least-squares fit. Sections require `--dm-segment-min-points 3` by default, while rejected short sections remain in the JSON and plot for auditability. The global fit follows the same weighting rule. Configure the automatic segmentation with:

```bash
psrism ARCHIVE_DIR --time-params dm \
  --dm-turning-sigma 2 --dm-segment-min-points 4
```

The accepted section slopes are combined as:

```math
\left\langle\left|\frac{dDM}{dt}\right|\right\rangle
=
\frac{\sum_j N_j\left|dDM/dt\right|_j}{\sum_j N_j},
```

where `N_j` is the number of measurements in section `j`, following the weighting described in the thesis. [3, PDF pp. 71-72, §4.7.2] PSRISM propagates independent fitted slope errors through that weighted mean when every accepted section has a finite slope error. Because shared turning points participate in both neighboring fits, their contribution is represented in both section weights; the JSON records `turning_point_overlap = true`.

The report intentionally does not fit the population relation `|Delta DM| approximately a DM^b`: the thesis presents `b = 0.5` from an earlier pulsar sample but states that applying that method to its own short, interval-dependent series is inappropriate. [3, PDF p. 72, §4.7.2, Eq. 4.9] It also does not infer mean electron density from `DM / D` because PSRISM has no independently validated pulsar-distance input in this workflow; the physical relation is given in the thesis. [3, PDF p. 71, §4.7.1, Eq. 4.8]

The generated products are:

```text
*_time_series_dm_solar.json
*_time_series_dm_vs_sun_separation.png
*_time_series_dm_piecewise_slopes.png
```

The JSON records the selected DM series, sample counts, warning threshold and count, ephemeris methods, Pearson result, global fit, every accepted or rejected section, segmentation settings, and the measurement-count-weighted absolute slope. The solar plot includes available DM error bars, the warning angle, and Pearson annotation. The slope plot shows the measurements, global trend, accepted section fits, and rejected short-section fits. No products are created when `dm` is excluded with `--time-params`.

Implementation:

```text
psrism/dm_variability.py
psrism/solar_geometry.py
psrism/time_series_analysis.py
psrism/cli.py
```

## Plotting

All heatmap-style plots use the `afmhot` color map. The plot title is the archive file name.

### Dynamic Spectrum

The dynamic spectrum shows pulse intensity as a function of observing time and radio frequency. It is the primary input for scintillation analysis. [1, §7.4.4.1]

For each subintegration and frequency channel, `psrism` estimates an on-pulse flux from the pulse area after off-pulse subtraction, following the standard dynamic-spectrum construction. [1, §7.4.4.1] PSRISM additionally normalizes by the off-pulse standard deviation and clips negative integrated values:

```math
E(t,\nu)=
\max\left[
0,\,
\frac{1}{N_{\rm on}}
\int_{\phi_{\rm on}}
\frac{P(t,\nu,\phi)-\mu_{\rm off}(t,\nu)}
{\sigma_{\rm off}(t,\nu)}
d\phi
\right].
```

Here `P(t,ν,φ)` is the pulse profile at time `t`, frequency `ν`, and pulse phase `φ`. The off-pulse mean and standard deviation are:

```math
\mu_{\rm off}(t,\nu), \qquad \sigma_{\rm off}(t,\nu).
```

In the code, the on-pulse window is estimated in [`psrism/dynamic_flux.py`](psrism/dynamic_flux.py). The dynamic spectrum is constructed in [`psrism/dynamic_spectrum.py`](psrism/dynamic_spectrum.py), and the plot with attached time and frequency marginals is made in [`psrism/plot_dynamic_spectrum.py`](psrism/plot_dynamic_spectrum.py).

Relevant functions:

```text
single_pulse_window
interpulse_windows
fluxes_single_peak
fluxes_two_peak
calculate_dynamic_spectrum
plot_dynamic_spectrum
```

Run:

```bash
psrism ARCHIVE --dspec
```

Example with scrunching:

```bash
psrism ARCHIVE --nsub 71 --nchan 183 --nbin 512 --dspec
```

The automatic on-pulse window keeps profile bins above `peak / width_factor`. The default `width_factor` is 10 and can be changed with `--onw`; the threshold rule and default are PSRISM choices:

```bash
psrism ARCHIVE --dspec --onw 15
```

For a pulsar with a distinct interpulse, `--interpulse 1` switches from one centered on-pulse window to two automatically identified windows; `--interpulse 0` is the default single-pulse mode. The two-window extension is a PSRISM implementation choice built on the same cited on-pulse integration. [1, §7.4.4.1]

```bash
psrism ARCHIVE --dspec --interpulse 1
```

`--normalize-dspec` optionally subtracts each frequency channel's time-mean bandpass and divides the result by the global finite-sample standard deviation after masking. This transform is a PSRISM choice and is applied consistently before requested dynamic-spectrum, ACF, secondary-spectrum, and directory scintillation calculations:

```bash
psrism ARCHIVE --dspec --acspec --sspec --normalize-dspec
```

The output file is:

```text
*_dspec.png
```

### Data Quality and RFI Masking

Every workflow that constructs a dynamic spectrum honors existing PSRCHIVE weights and rejects non-finite samples. These checks are always active. Statistical RFI detection is deliberately opt-in:

```bash
psrism ARCHIVE --dspec --rfi-mask
```

The reference analyses use automated median/spike excision followed by manual or visual RFI inspection. [3, PDF p. 29, §2.6.1; 7, PDF p. 2, §2] PSRISM implements its own reproducible robust-median/MAD detector on the processed archive dimensions, after requested time, frequency, and phase scrunching, to identify three contamination patterns:

- persistent narrow-band contamination from anomalous channel level or spread;
- impulsive broad-band contamination from anomalous time-bin level or spread;
- isolated time-frequency outliers after subtracting robust time and frequency trends.

The default threshold of six robust standard deviations and the minimum valid fraction of one half are PSRISM choices, not values prescribed by the references. Both controls are explicit:

```bash
psrism ARCHIVE --dspec --rfi-mask \
  --rfi-sigma 6 \
  --rfi-min-valid-fraction 0.5
```

When `--rfi-mask` is active, rejected samples have their PSRCHIVE weights set to zero before requested integrated-profile, subband tau, alpha, or anisotropic fits are performed. The same mask is then used for the dynamic spectrum, ACF, and secondary spectrum. Masked pixels are shown in gray in the dynamic-spectrum plot.

For masked ACF calculations, PSRISM corrects each lag by the number of overlapping valid sample pairs before normalizing at zero lag. For the secondary spectrum, the mean of the valid samples is removed and masked samples are set to zero only after mean subtraction. This prevents rejected values and NaNs from entering the Fourier transform without inserting a non-zero artificial signal.

Single-archive runs save an audit record:

```text
*_data_quality.json
```

It records the processed shape, masking mode, thresholds, invalid-cell count, rejected time-bin and channel indices, isolated-cell count, and total masked fraction. Directory-mode CSV output includes:

```text
quality_masked_fraction
quality_bad_time_bins
quality_bad_frequency_channels
quality_isolated_flagged_cells
```

Directory CSV files also record `profile_intrinsic_model`, `profile_components`, and `intrinsic_template` so later tau and alpha comparisons retain the profile-model provenance.

The terminal always states whether only archive weights were honored or statistical masking was enabled. Automatic masking is intentionally conservative, but it is not a replacement for visual inspection or observatory-specific RFI excision. Bright compact scintles can resemble isolated outliers, so important measurements should compare the quality report and dynamic spectrum with and without `--rfi-mask`. Increasing `--rfi-sigma` makes the detector less aggressive.

### Autocorrelation Spectrum

The autocorrelation spectrum measures how similar the dynamic spectrum is to itself after a time lag and a frequency lag. It is used to estimate the characteristic diffractive scintillation bandwidth and timescale. [1, §7.4.4.1]

Following the Lorimer-Kramer convention, the finite-lag covariance is: [1, §7.4.4.1, Eq. 7.37]

```math
{\rm CF}(\Delta\nu,\Delta t)=
\sum_{\nu=1}^{n_\nu-|\Delta\nu|}
\sum_{t=1}^{n_t-|\Delta t|}
E(\nu,t)\,
E(\nu+\Delta\nu,t+\Delta t).
```

The normalized autocorrelation function is: [1, §7.4.4.1, Eq. 7.38]

```math
{\rm ACF}(\Delta\nu,\Delta t)=
\frac{{\rm CF}(\Delta\nu,\Delta t)}
{{\rm CF}(0,0)}.
```

In the code, the mean dynamic-spectrum level is subtracted before computing the covariance. The calculation uses zero-padded linear correlation, not circular FFT autocorrelation. This is implemented in [`psrism/autocorrelation_spectrum.py`](psrism/autocorrelation_spectrum.py) by `calculate_covariance_function` and `calculate_autocorrelation_spectrum`. Plotting is handled in [`psrism/plot_autocorrelation_spectrum.py`](psrism/plot_autocorrelation_spectrum.py).

Relevant functions:

```text
calculate_covariance_function
calculate_autocorrelation_spectrum
autocorrelation_axes
measure_acf_scales
fit_autocorrelation_spectrum
plot_autocorrelation_spectrum
```

Run:

```bash
psrism ARCHIVE --acspec
```

This saves the full unzoomed lag window. To make the fitted central ellipse easier to inspect, use:

```bash
psrism ARCHIVE --acspec --zoom-acf
```

The zoomed view keeps the coordinate axes and fit overlays, but crops the heatmap around the fitted central ACF lobe.

The output file is:

```text
*_acspec.png
```

For `--acspec --zoom-acf`, the output file is:

```text
*_acspec_zoom.png
```

When `--acspec` is used, `psrism` also measures the two characteristic scintillation scales and writes the full audit record to:

```text
*_scintillation.json
```

The decorrelation bandwidth is measured from the zero-time-lag frequency cut: [1, §7.4.4.1]

```math
\Delta\nu_{\rm DISS}:
{\rm ACF}(\Delta\nu,0)=\frac{1}{2}.
```

The diffractive scintillation timescale is measured from the zero-frequency-lag time cut: [1, §7.4.4.1]

```math
\Delta t_{\rm DISS}:
{\rm ACF}(0,\Delta t)=\frac{1}{e}.
```

The Handbook recommends obtaining these widths by fitting a two-dimensional Gaussian to the ACF. [1, §7.4.4.1] The reference LOFAR monitoring work likewise obtains both scintillation scales from the 2D ACF and notes that their statistical uncertainties can be underestimated when frequency resolution is limited. [7, PDF p. 2, §1; PDF p. 4, §3.2]

PSRISM therefore uses the tilted 2D Gaussian fit as its primary measurement. The direct positive-lag slice crossings remain independent cross-checks and provide a coverage test: if the relevant slice never reaches 1/2 or 1/e inside the computed lag window, the corresponding fit is retained in the report but is marked unresolved and is not published as an accepted scale.

It also requires each fitted width to span at least two native dynamic-spectrum samples. This two-sample threshold is a conservative PSRISM quality choice, not a value prescribed by the references. Change it explicitly when justified by the data:

```bash
psrism ARCHIVE --acspec --scintillation-min-resolution-bins 3
```

For frequency, the native sample is `bandwidth / nchan`; for time it is `observation duration / nsub`. A scale is accepted only when both its defining slice crossing exists and:

```math
\frac{\Delta\nu_{\rm DISS}}{\delta\nu_{\rm channel}}
\geq N_{\rm min},
\qquad
\frac{\Delta t_{\rm DISS}}{\delta t_{\rm subint}}
\geq N_{\rm min}.
```

These ratios are resolution diagnostics defined by PSRISM. They do not replace visual inspection of the dynamic spectrum and fitted ACF, as performed in the reference LOFAR workflow. [7, PDF p. 3, §2]

The width, correlation, and drift uncertainties are first-order propagation of SciPy's local least-squares covariance through the fitted Gaussian equations. This propagation is a PSRISM choice. Because neighboring ACF pixels are correlated, these values are local fit errors rather than complete systematic uncertainties; finite observing coverage, residual RFI, profile nulling, and imperfect ACF shape can make the true uncertainty larger.

When both scales are accepted, PSRISM also estimates the number of scintles sampled by the observation: [1, §4.2.5.2; 5, PDF p. 5, Eqs. 6-7]

```math
N_{\rm scintles}
=
\left(1+\eta_t\frac{T}{\Delta t_{\rm DISS}}\right)
\left(1+\eta_\nu\frac{B}{\Delta\nu_{\rm DISS}}\right),
\qquad
\epsilon_{\rm finite}=\frac{1}{\sqrt{N_{\rm scintles}}}.
```

The default filling factors are `eta_t = eta_nu = 0.2`, and `N_scintles` is bounded below by one following the cited prescription. [1, §4.2.5.2; 5, PDF p. 5, Eq. 7] `--eta-time` and `--eta-freq` change them. The reported `finite_scintle_fraction` is an observing-coverage diagnostic; PSRISM does not add it to the 2D Gaussian width errors because the cited `tau / sqrt(N_scintles)` prescription concerns the finite-scintle contribution to scattering-delay uncertainty, not a universal error law for both ACF widths. [5, PDF p. 5, Eq. 6]

If the 2D fit fails, PSRISM falls back to the interpolated slice crossings, labels the method `acf_slice_crossing`, and leaves covariance-error fields empty. Otherwise the method is `tilted_gaussian_2d`. Terminal output includes the accepted values, errors, resolution ratios, slice cross-checks, and explicit unresolved warnings, for example:

```text
ACF scintillation measurement:
 primary method = tilted_gaussian_2d
 frequency resolution = ... MHz; minimum accepted width = 2 samples
 decorrelation bandwidth = ... +/- ... MHz (... samples)
 time resolution = ... s
 diffractive timescale = ... +/- ... s (... samples)
 slice cross-checks: dnu_d=... MHz, dt_d=... s
```

The JSON report stores the accepted values and errors together with raw fit values, slice values, drift and correlation measurements, native resolutions, resolution ratios, slice-crossing flags, resolved flags, method, and configured minimum width. Directory CSV output stores the same provenance in these columns:

```text
decorrelation_bandwidth_mhz
decorrelation_bandwidth_error_mhz
diffractive_timescale_s
diffractive_timescale_error_s
scintillation_measurement_method
acf_fit_decorrelation_bandwidth_mhz
acf_fit_decorrelation_bandwidth_error_mhz
acf_fit_diffractive_timescale_s
acf_fit_diffractive_timescale_error_s
acf_fit_width_covariance_mhz_s
acf_width_covariance_mhz_s
acf_slice_decorrelation_bandwidth_mhz
acf_slice_diffractive_timescale_s
acf_drift_slope_s_per_mhz
acf_drift_slope_error_s_per_mhz
acf_correlation
acf_correlation_error
scintillation_frequency_resolution_mhz
scintillation_time_resolution_s
decorrelation_bandwidth_resolution_bins
diffractive_timescale_resolution_bins
decorrelation_bandwidth_resolved
diffractive_timescale_resolved
scintillation_n_scintles
scintillation_finite_scintle_fraction
scintillation_eta_time
scintillation_eta_frequency
refractive_timescale_error_days
```

Only accepted values enter `dnu_d`, `dt_d`, their time-series plots, and the derived `t_r` calculation.

#### Tilted ACF and Scintle Drift

In an ideal stationary diffractive scintillation pattern, the central ACF lobe is roughly elliptical and aligned with the time-lag and frequency-lag axes. A tilted central ellipse means that scintillation maxima at one frequency are correlated with maxima at another frequency after a shifted time lag. Drift bands are associated with a refractive angular offset in the scattering pattern. [1, §4.2.5.3]

The physical picture is that a refractive phase gradient displaces the apparent scattered image. Because the displacement is chromatic, the diffraction pattern shifts with observing frequency. If this shift has a component parallel to the effective transverse velocity, the dynamic spectrum shows tilted scintillation bands and the ACF becomes skewed. [1, §4.2.5.3]

Standard scintillation analysis fits a two-dimensional Gaussian to the ACF. [1, §7.4.4.1] PSRISM uses the following tilted, positive-definite parameterization:

```math
C(\Delta t,\Delta\nu)=
C_{\rm bg}
+
C_0
\exp\left[
-
\left(
a\Delta t^2
+
b\Delta t\Delta\nu
+
c\Delta\nu^2
\right)
\right].
```

The fitted ridge and the coordinate-axis widths below follow algebraically from PSRISM's Gaussian parameterization:

```math
\frac{d\Delta t}{d\Delta\nu}
=
-
\frac{b}{2a}.
```

This slope is reported as the scintle drift slope in seconds per MHz. The inverse slope is also printed as MHz per second. The same fit gives model-based coordinate-axis estimates of the diffractive timescale and decorrelation bandwidth:

```math
\Delta t_{\rm DISS,fit}
=
\frac{1}{\sqrt{a}},
\qquad
\Delta\nu_{\rm DISS,fit}
=
\sqrt{\frac{\ln 2}{c}}.
```

The implementation is in [`psrism/fit_autocorrelation_spectrum.py`](psrism/fit_autocorrelation_spectrum.py). The fit uses a bounded correlation-coefficient parameterization so the fitted ellipse remains positive definite, then converts the result to the quadratic coefficients `a`, `b`, and `c`. [`psrism/plot_autocorrelation_spectrum.py`](psrism/plot_autocorrelation_spectrum.py) overlays the fitted half-power contour and the drift ridge on the ACF plot.

The optimizer uses the central 35 per cent of each lag axis and, when at least 20 samples remain, restricts that window to values above the larger of 0.03 or 10 per cent of the ACF peak. It bounds each Gaussian scale below by one lag sample, limits the correlation coefficient to `[-0.95, 0.95]`, uses at most 50,000 deterministically spaced fit samples, and permits 20,000 function evaluations. These are PSRISM fit-stability choices, not constants from the references.

The terminal output includes:

```text
Tilted ACF Gaussian fit:
 fit decorrelation bandwidth = ... +/- ... MHz
 fit diffractive timescale = ... +/- ... s
 correlation coefficient = ... +/- ...
 ellipse rotation angle = ... deg
 scintle drift slope d(time lag)/d(freq lag) = ... +/- ... s/MHz
 inverse drift rate d(freq lag)/d(time lag) = ... MHz/s
 quadratic coefficients: a=..., b=..., c=...
 goodness: unweighted reduced chi-square = ..., RMS residual = ..., fit points = ...
```

### Secondary Spectrum

The secondary spectrum is formed from the two-dimensional Fourier transform of the dynamic spectrum and is viewed in conjugate frequency and conjugate time. [1, §7.4.4.3] PSRISM uses the squared magnitude as its linear scientific power.

Mean subtraction is a PSRISM preprocessing choice:

```math
E'(t,\nu)=E(t,\nu)-\langle E\rangle.
```

Then PSRISM computes Fourier power from the reference definition: [1, §7.4.4.3]

```math
S(f_t,f_\nu)=
\left|
{\cal F}_2\left\{E'(t,\nu)\right\}
\right|^2.
```

By default, PSRISM applies a separable Hann taper before the transform to reduce power leaked from the edges of a finite dynamic spectrum. This is a PSRISM signal-processing choice rather than a method specified by the bundled references. The taper is normalized by its RMS over valid samples so its effect on the overall linear-power scale is limited. An axis with fewer than three samples is left untapered because a conventional Hann window would contain no useful interior samples. Disable tapering explicitly with:

```bash
psrism ARCHIVE --sspec --secondary-window none
```

For plotting, `psrism` uses the following PSRISM display choice; the `10^-12` floor only avoids `log10(0)`:

```math
S_{\rm dB}=10\log_{10}\left(S+10^{-12}\right).
```

The FFT-shifted axes are made with `numpy.fft.fftfreq`. The x-axis is fringe frequency in Hz. The y-axis is delay in seconds because the frequency-channel spacing is converted from MHz to Hz before building the Fourier-conjugate axis. For an observation of duration `T`, bandwidth `B`, time sample `delta_t`, and channel width `delta_nu`, PSRISM reports the following discrete-Fourier sampling diagnostics:

```math
\delta f_{\rm D}=\frac{1}{T},
\qquad
\delta\tau=\frac{1}{B},
\qquad
f_{{\rm D,Nyq}}=\frac{1}{2\delta t},
\qquad
\tau_{\rm Nyq}=\frac{1}{2\delta\nu}.
```

These diagnostics follow directly from the discrete Fourier grid and are PSRISM bookkeeping, while the physical conjugate-axis interpretation follows the Handbook. [1, §7.4.4.3]

The returned scientific spectrum is no longer modified to hide its central row or column. For the image and arc search only, `--arc-mask-bins` masks the specified number of bins around both zero fringe frequency and zero delay, where residual DC and axis leakage can dominate. This separation is a PSRISM implementation choice: the JSON metadata and calculations remain explicit about the mask while the underlying Fourier power stays intact.

The calculation is implemented in [`psrism/scintillation_spectrum.py`](psrism/scintillation_spectrum.py), and plotting with attached marginals is implemented in [`psrism/plot_scintillation_spectrum.py`](psrism/plot_scintillation_spectrum.py).

Relevant functions:

```text
calculate_scintillation_spectrum
fit_parabolic_arc
plot_scintillation_spectrum
```

Run:

```bash
psrism ARCHIVE --sspec
```

The output file is:

```text
*_sspec.png
```

Every `--sspec` or `--fit-arc` run also prints the FFT configuration and sampling limits:

```text
Secondary spectrum sampling:
 FFT window = hann
 masked dynamic-spectrum fraction = ...
 dynamic-spectrum sampling = ... s x ... MHz
 Fourier resolution = ... Hz x ... s
 Nyquist limits = +/-... Hz, +/-... s
```

#### Parabolic Arc Fitting

Secondary spectra often show parabolic scintillation arcs. Their detection supports localized thin-screen structures, while multiple arcs can indicate multiple scattering screens. [2, PDF pp. 1-2, §1; PDF p. 9, §4] In the simple thin-screen model, the arc curvature depends on screen position, wavelength, and scintillation velocity. [1, §7.4.4.3, Eq. 7.45] A useful route to the relation is that geometric delay scales as:

```math
\tau_{\rm d}
\propto
\theta^2,
```

while the Doppler or fringe frequency scales approximately as: [1, §7.4.4.3]

```math
f_{\rm D}
\propto
\theta.
```

Eliminating `θ` gives the parabolic arc relation. [1, §7.4.4.3]

```math
\tau_{\rm d}
=
\eta f_{\rm D}^2.
```

The curvature `η` is the fitted observable. In a simple thin-screen geometry it is related to effective distance, observing wavelength, and effective transverse velocity by: [1, §7.4.4.3, Eq. 7.45]

```math
\eta
=
\frac{
D_{\rm eff}\lambda^2
}{
2cV_{\rm eff,\parallel}^2
}.
```

This relation is physically useful, but it is model-dependent: the Handbook equation assumes a simple thin screen and a scintillation pattern dominated by the quoted scintillation velocity. [1, §7.4.4.3]

At fixed geometry and effective velocity, the `lambda^2` term implies `eta` scales as observing frequency to the power `-2`. The bundled LOFAR analysis uses this scaling when comparing measured and predicted arc curvature, and notes that disagreement may indicate multiple screens. [7, PDF p. 4, §3.2]

In `psrism`, arc fitting is requested with:

```bash
psrism ARCHIVE --sspec --fit-arc
```

The fitted model is PSRISM's offset generalization of the cited parabola:

```math
\tau_{\rm d}
-
\tau_0
=
\eta
\left(
f_{\rm D}
-
f_{{\rm D},0}
\right)^2.
```

By default, the apex offsets are zero as a PSRISM configuration choice:

```math
f_{{\rm D},0}=0,
\qquad
\tau_0=0.
```

Explicit offsets can be supplied when testing offset arclets or lens-like structures:

```bash
psrism ARCHIVE --sspec --fit-arc \
  --arc-fringe-offset 0.002 \
  --arc-delay-offset 1e-6
```

The code uses a PSRISM arc-strength search; the bundled references motivate a parabola but do not prescribe this estimator. For each trial curvature, it samples the linear secondary-spectrum power along the parabola and computes:

```math
A(\eta)
=
\frac{1}{N_\eta}
\sum_{i=1}^{N_\eta}
S_2
\left[
f_{{\rm D},i},
\tau_0
+
\eta
\left(
f_{{\rm D},i}
-
f_{{\rm D},0}
\right)^2
\right].
```

The best raw curvature is the trial that maximizes `A(η)`. As PSRISM choices, the descriptive uncertainty interval is bounded by the two half-height crossings above the median arc-strength baseline, and the central DC spike and axis leakage are masked before scoring. This width is not a formal confidence interval.

Unless explicit bounds are supplied, PSRISM chooses a logarithmic search range from the resolvable Fourier grid. Its shallow limit reaches the first nonzero delay at the largest unmasked fringe frequency, while its steep limit reaches the largest delay at the first unmasked fringe-frequency bin:

```math
\eta_{\rm min}
=
\frac{\delta\tau}{\max|f_{\rm D}|^2},
\qquad
\eta_{\rm max}
=
\frac{\max|\tau|}{\min|f_{\rm D,unmasked}|^2}.
```

This automatic range and its logarithmic grid are PSRISM search choices. `--arc-curvature-min`, `--arc-curvature-max`, and `--arc-curvature-trials` override the range or grid density.

PSRISM separately decides whether that raw maximum is accepted. By default an accepted curvature must:

- have at least 20 sampled arc points;
- have a robust arc-strength-curve S/N of at least 5;
- have both half-height crossings inside the searched curvature range; and
- not place its maximum at either search boundary.

These are conservative PSRISM quality choices, not thresholds prescribed by the references. The score S/N uses the median trial score as its baseline and `1.4826 * MAD` as its noise scale, falling back to the standard deviation only when the MAD is zero. Trial scores are correlated, so this is a search diagnostic rather than a calibrated detection significance. A rejected fit retains its raw curvature, search curve, and explicit rejection reasons, but `accepted_curvature` is empty.

Useful options are:

```bash
psrism ARCHIVE --sspec --fit-arc --arc-half positive
psrism ARCHIVE --sspec --fit-arc --arc-half both
psrism ARCHIVE --sspec --fit-arc --arc-curvature-trials 400
psrism ARCHIVE --sspec --fit-arc --arc-curvature-min 1e-3 --arc-curvature-max 1e2
psrism ARCHIVE --sspec --fit-arc --arc-mask-bins 4
psrism ARCHIVE --fit-arc --arc-min-samples 30 --arc-min-score-snr 6
psrism ARCHIVE --sspec --secondary-window hann
```

The implementation is in [`psrism/fit_secondary_spectrum.py`](psrism/fit_secondary_spectrum.py). The secondary-spectrum plot overlays the fitted parabola when `--sspec` and `--fit-arc` are used together; accepted fits are cyan and rejected raw fits are gray.

Every successful arc search writes:

```text
*_arc_fit.json
*_arc_search.png
```

The JSON report contains the raw and accepted curvature, signed half-height bounds, status and rejection reasons, score baseline and robust noise, score S/N, sampled-point count, configured thresholds, apex offsets, delay half, boundary flag, complete trial-curvature and arc-strength arrays, observing center frequency, FFT window, masked fraction, native dynamic-spectrum sampling, Fourier-axis resolution, and Nyquist limits. The diagnostic plot shows the complete search curve, its median baseline, the best raw curvature, its half-height interval when closed, and whether the result was accepted. `--fit-arc` produces these two files even without `--sspec`.

Terminal output includes:

```text
Parabolic arc fit:
 status = accepted/rejected
 delay half searched = positive
 apex fringe-frequency offset = 0 Hz
 apex delay offset = 0 s
 curvature eta = ... +/- ... s^3
 half-height interval = [..., ...] s^3
 arc strength = ...
 arc strength S/N = ...
 samples on best arc = ... (minimum ...)
 peak at search boundary = ...
 rejection reasons = ...
```

### Integrated Profile

The integrated profile shows the pulse profile after integration over observing time and frequency. It is useful for visualizing the mean pulse shape and for estimating pulse broadening.

The package also shows frequency-versus-phase and time-versus-phase panels, sharing the same pulse phase axis. Time is shown in minutes.

Run:

```bash
psrism ARCHIVE --intpf
```

When `--intpf` is used, the integrated profile is fit with the scatter-broadened model described in the fitting section. The model is overlaid on the integrated-profile panel. The figure caption reports the fitted τ, its uncertainty, and unweighted fit-quality statistics.

The plot is implemented in [`psrism/plot_dynamic_spectrum.py`](psrism/plot_dynamic_spectrum.py). The model fit is implemented in [`psrism/fit_tau.py`](psrism/fit_tau.py).

Relevant functions:

```text
plot_integrated_profile
fit_tau_from_archive
fit_tau_from_profile
scattered_pulse
```

The output file is:

```text
*_intpf.png
```

Terminal output includes:

```text
Integrated-profile tau fit:
 intrinsic model = gaussian
 Gaussian components = 1
 fitted DC baseline = ...
 tau (bins) = ...
 tau = ... +/- ... s
 goodness: unweighted reduced chi-square = ..., RMS residual = ...
```

## Fitting

### Scattering Timescale τ

The scattering timescale τ describes characteristic pulse broadening caused by multipath propagation through the ionized interstellar medium. PSRISM follows the train+DC forward-fitting approach: a periodic intrinsic pulse train is convolved with a causal pulse-broadening function, and a constant baseline is fitted with the physical parameters. This preserves scattering power that wraps from preceding rotations. [3, PDF p. 33, §2.7.2; 4, PDF pp. 7-8, §2.4]

For isotropic thin-screen scattering, the causal pulse-broadening function is: [4, PDF p. 3, Eq. 1; 2, PDF p. 4, thin-screen PBF paragraph]

```math
g(t)=\frac{1}{\tau}\exp\left(-\frac{t}{\tau}\right)U(t).
```

PSRISM folds the infinite pulse train into one rotation. For phase-bin delay `j` and `N` bins per rotation, the discrete normalized kernel is:

```math
g_P[j]
=
\frac{\exp(-j/\tau)}
{\sum_{k=0}^{N-1}\exp(-k/\tau)},
\qquad 0\le j<N.
```

This expression is PSRISM's discrete implementation of the cited exponential PBF and train method. The fitted profile is:

```math
P[j]
=
C
+
I[j]\circledast g_P[j],
```

where `C` is the fitted DC level, `I` is the intrinsic profile, and `circledast` is circular convolution over one rotation. The default intrinsic profile is one Gaussian. A sum of `K` Gaussian components is:

```math
I(\phi)
=
\sum_{k=1}^{K}
A_k
\exp\left[
-\frac{d_P(\phi,\mu_k)^2}{2\sigma_k^2}
\right],
```

where `d_P` is the shortest periodic phase distance. Gaussian intrinsic profiles, sums of Gaussians for multicomponent profiles, templates from minimally scattered high-frequency observations, and a fitted DC level follow the forward-fitting strategy in the references. [2, PDF p. 4, §2.2; 3, PDF pp. 32-34, §§2.7.1-2.7.3; 4, PDF pp. 7-8, §2.4] The periodic-distance parameterization, multistart optimizer, parameter bounds, and covariance-based errors are PSRISM implementation choices. For numerical conditioning, PSRISM subtracts the whole-profile median and divides by the peak-to-peak range before fitting; because DC remains free, this affine transformation does not perform off-pulse baseline estimation or remove wrap-around structure.

Implementation:

```text
psrism/fit_tau.py
```

The main functions are `isotropic_scattered_pulse`, `fit_tau_from_profile`, `fit_tau_from_archive`, and `evaluate_tau_fit`. The older `scattered_pulse` analytic EMG is retained as a library compatibility helper, but archive fitting now uses the periodic train+DC model.

Run an integrated-profile fit with either:

```bash
psrism ARCHIVE --intpf
```

or:

```bash
psrism ARCHIVE --fit-tau
```

`--intpf` saves the fit overlay in:

```text
*_intpf.png
```

`--fit-tau` saves a separate fit-only plot:

```text
*_tau_fit.png
```

Use more Gaussian components when the intrinsic profile is visibly multicomponent:

```bash
psrism ARCHIVE --fit-tau --profile-components 2
psrism ARCHIVE --fit-alpha --tau-subbands 6 --profile-components 2
```

Alternatively, provide an intrinsic high-frequency or otherwise minimally scattered template:

```bash
psrism ARCHIVE --fit-tau --intrinsic-template TEMPLATE.npy
psrism ARCHIVE --fit-alpha --tau-subbands 6 --intrinsic-template TEMPLATE.ar
```

Templates may be one-dimensional NPY, text, or CSV data, or a PSRCHIVE-readable archive. PSRISM baseline-removes and normalizes the template, resamples it periodically to the observed bin count when needed, and fits its amplitude and fractional phase shift. `--intrinsic-template` and a `--profile-components` value other than 1 are mutually exclusive because they represent alternative intrinsic-profile models.

### Scattering Spectral Index α

The frequency scaling of scattering is modeled with positive `alpha`: [1, §4.2.3; 3, PDF pp. 34-35, Eq. 2.6]

```math
\tau(\nu)=
\tau_{\rm ref}
\left(
\frac{\nu}{\nu_{\rm ref}}
\right)^{-\alpha}.
```

In logarithmic form, equivalent to the thesis fit: [3, PDF pp. 34-35, Eq. 2.6]

```math
\log_{10}\tau =
-\alpha
\log_{10}
\left(
\frac{\nu}{\nu_{\rm ref}}
\right)
+
\log_{10}\tau_{\rm ref}.
```

For a single archive, `psrism` divides the full bandwidth into contiguous frequency subbands. This follows the reference workflow, where the number of subbands is chosen as a compromise between frequency sampling and profile S/N. [3, PDF p. 31, §2.6.3; 2, PDF p. 3, §2.2]

A fitted profile represents a finite-width subband rather than a monochromatic observation. PSRISM first records the arithmetic band center `f_c` and full subband width `delta_f`. [4, PDF p. 4, Eqs. 5-6] Including the outer half-channel edges when calculating the full width is a PSRISM implementation choice:

```math
f_c =
\frac{f_{\rm min}+f_{\rm max}}{2}.
```

It then assigns the measured τ to the effective monochromatic frequency `f_m`: [4, PDF p. 4, Eq. 6]

```math
f_m =
\frac{1}{2}
\sqrt{
\delta_f^2 + 4f_c^2
}.
```

Using `f_m` avoids the spectral-index bias demonstrated when finite-band measurements are plotted at `f_c`. [4, PDF pp. 4-5, Eqs. 5-6 and Fig. 7] Terminal output reports `fc`, `bandwidth`, and `fm` separately so this correction remains auditable.

Each subband profile is fit independently to obtain:

```math
(\tau_i,\sigma_{\tau_i}).
```

Only subbands that pass the quality rules below enter the frequency-scaling fit. PSRISM performs an analytic weighted fit in log space; the fitted physical relation is the cited scattering power law. [3, PDF pp. 34-35, Eq. 2.6]

```math
x_i =
\log_{10}
\left(
\frac{\nu_i}{\nu_{\rm ref}}
\right),
\qquad
y_i =
\log_{10}(\tau_i),
```

with propagated uncertainty:

```math
\sigma_{y_i} =
\frac{\sigma_{\tau_i}}
{\tau_i\ln 10}.
```

The fitted line is:

```math
y_i = -\alpha x_i + b,
```

and:

```math
\tau_{\rm ref} = 10^b.
```

The reference analysis estimates the α uncertainty both from least squares and Monte Carlo sampling, then retains the larger estimate. [2, PDF p. 4, §2.2; 3, PDF pp. 34-35, §2.7.4] PSRISM follows that procedure. By default it makes 10,000 reproducible split-normal draws in log τ using random seed 0, refits each draw with the same weights, and reports the 16th-to-84th-percentile interval for both α and `tau_ref`. Split-normal log-space sampling, 10,000 draws, and seed 0 are PSRISM choices rather than values prescribed by the references.

The result contains both asymmetric Monte Carlo intervals and symmetric covariance errors. The compatibility fields `alpha_error` and `tau0_error` conservatively contain the largest corresponding covariance, lower-percentile, or upper-percentile error. Direct Python calls to `fit_alpha` may supply symmetric `tau_error` with shape `(N,)` or separate lower and upper errors with shape `(2, N)`.

Implementation:

```text
psrism/fit_tau.py
psrism/fit_alpha.py
psrism/plot_tau_vs_freq.py
```

Run:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 4
```

The paper rejects weak or interference-dominated subbands and requires at least three reliable τ measurements for a frequency-scaling fit, preferring four where possible. [2, PDF p. 5, §3] PSRISM makes that decision reproducible with these default quality rules:

- fitted τ and its uncertainty must be finite and positive;
- residual-based profile-fit S/N must be at least 5;
- `tau_error / tau` must not exceed 1;
- at least half of the subband channels must retain positive PSRCHIVE weights;
- at least three subbands must pass all rules.

The numerical thresholds are PSRISM project defaults, because the references use data-dependent visual quality decisions rather than prescribing universal cutoffs.

The residual-based profile-fit S/N is a PSRISM diagnostic, defined from the fitted model `M` and residual `r = I - M` as:

```math
{\rm S/N}_{\rm fit}
=
\frac{\max(M)-\min(M)}
{1.4826\,{m median}\left|r-{m median}(r)\right|}.
```

The factor 1.4826 scales the median absolute deviation to the standard deviation for Gaussian noise; its use and this complete diagnostic definition are PSRISM choices. Change the thresholds explicitly when the source, telescope, or preprocessing warrants it:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 6 \
  --alpha-min-subbands 4 \
  --alpha-min-profile-snr 7 \
  --alpha-max-relative-tau-error 0.5 \
  --alpha-min-valid-channel-fraction 0.7
```

Every successfully fitted subband is marked `accepted` or `rejected`, with all rejection reasons printed. A non-convergent subband is reported separately as `fit failed`. If fewer than `--alpha-min-subbands` survive, the single-archive command stops without publishing an α value; directory mode skips that epoch and records the reason in its terminal log.

Monte Carlo sample count and seed are configurable. Set the sample count to zero to use weighted least-squares covariance errors only:

```bash
psrism ARCHIVE --fit-alpha \
  --alpha-mc-samples 20000 \
  --alpha-mc-seed 42

psrism ARCHIVE --fit-alpha --alpha-mc-samples 0
```

The default reference frequency is 150 MHz, matching the thesis and PL611 analysis convention, so `tau_ref` is reported as `tau150`. [3, PDF p. 35, §2.7.4; 2, PDF p. 4, §2.2] To use a different reference frequency:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 4 --tau-reference-freq 1000
```

Terminal output includes subband τ measurements and the fitted α:

```text
Subband tau fits:
 channels 0-90: fc=... MHz, bandwidth=... MHz, fm=... MHz, tau=... +/- ... s
  intrinsic model = gaussian, DC baseline = ..., fit S/N = ..., valid channels = ..., status = accepted
 channels 91-181: fc=... MHz, bandwidth=... MHz, fm=... MHz, tau=... +/- ... s
  intrinsic model = gaussian, DC baseline = ..., fit S/N = ..., valid channels = ..., status = rejected
  rejection reason(s): ...

alpha = ... +.../-... (adopted +/- ...)
alpha convention: tau(nu) = tau_ref * (nu / nu_ref)^(-alpha)
tau150 = ... +.../-... s (adopted +/- ... s) at nu_ref=150 MHz
uncertainty = weighted_least_squares+monte_carlo; Monte Carlo samples = ...
```

Directory-mode CSV files retain the compatibility fields `alpha_error` and `tau0_error_s` as the adopted conservative errors. They also store `alpha_error_lower`, `alpha_error_upper`, `alpha_covariance_error`, `alpha_monte_carlo_samples`, `tau0_error_lower_s`, `tau0_error_upper_s`, and accepted/rejected/failed subband counts. The meaning of `tau0_s` remains explicit in the adjacent `tau_reference_frequency_mhz` field.

This command saves:

```text
*_tau_vs_freq.png
*_subband_tau_fits.png
```

The τ-versus-frequency plot distinguishes accepted points from rejected points, shades the Monte Carlo 68 per cent model interval, and includes a comparison line with fixed `α = 4.4`, the Kolmogorov-spectrum expectation stated in the Handbook, normalized to the fitted `tau_ref` at the same reference frequency. [1, §4.2.3] Its global caption reports weighted χ², reduced χ², degrees of freedom, RMS log residual, fitted-point count, and Monte Carlo draw count. The subband-profile plot marks rejected fits with dashed gray curves and prints fit S/N and channel occupancy.

The subband fit overlay plot uses an automatic compact grid when no layout is supplied. For prime numbers of subbands, empty panels are hidden. You can choose the grid manually with:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 10 \
  --subband-plot-rows 2 \
  --subband-plot-cols 5
```

If only one grid dimension is supplied, `psrism` computes the other one. The same grid options are used by `--fit-anisotropy`.

### Intrinsic Profile Evolution

The observed pulse profile is a convolution of the intrinsic profile and the scattering pulse-broadening function. The reference analysis therefore allows the fitted intrinsic Gaussian width to change with observing frequency and plots its FWHM as a percentage of the pulse period. Such changes can represent radius-to-frequency mapping or natural profile broadening, but they can also be confused with scattering by the fit. [2, PDF pp. 11-12, Appendix B; PDF p. 12, Fig. B1]

PSRISM now retains the covariance uncertainty of every fitted Gaussian `sigma` in every successfully fitted subband. It converts each result using the standard Gaussian identity

```math
W_{\rm FWHM}=2\sqrt{2\ln 2}\,\sigma,
```

and reports the width in phase bins, pulse phase, percentage of pulse period, and seconds. The literature source defines and plots the intrinsic Gaussian FWHM; the algebraic conversion, covariance propagation, and additional units are PSRISM implementation choices. [2, PDF pp. 11-12, Appendix B; PDF p. 12, Fig. B1]

For each phase-ordered Gaussian component with enough accepted subbands, PSRISM also fits the diagnostic relation

```math
W(\nu)=W_{\rm ref}
\left(\frac{\nu}{\nu_{\rm ref}}\right)^{\gamma}.
```

The power law, weighted log-linear estimator, Student-t test of `gamma = 0`, default requirement of three accepted subbands, and default `p = 0.05` threshold are PSRISM choices; Appendix B supplies the width-versus-frequency diagnostic but does not prescribe this trend model or these cutoffs. A significant trend is printed as a caution, not classified as proof of radius-to-frequency mapping or of a scattering bias.

Intrinsic-width analysis runs automatically with either subband scattering command:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 6
psrism ARCHIVE --fit-anisotropy --tau-subbands 6
```

The defaults can be changed explicitly:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 8 \
  --intrinsic-width-min-subbands 4 \
  --intrinsic-width-p-threshold 0.01
```

Isotropic fitting saves:

```text
*_intrinsic_width.json
*_intrinsic_width_vs_freq.png
```

Anisotropic fitting saves separate products:

```text
*_anisotropic_intrinsic_width.json
*_anisotropic_intrinsic_width_vs_freq.png
```

The strict-JSON reports retain accepted and rejected measurements, subband frequency metadata, width uncertainties and units, fitted trend statistics, component-matching convention, thresholds, and interpretation cautions. Rejected scattering subbands remain visible in the plot but do not enter the trend fit.

For `--profile-components N`, components are numbered by fitted phase order independently in each subband. This is a PSRISM matching choice; crossing, phase wrapping, or blended components can exchange identities, so the subband profile overlays must be inspected before interpreting a component trend. An `--intrinsic-template` fit has no fitted Gaussian `sigma`; PSRISM writes a JSON report marking intrinsic width unavailable and does not create a width plot rather than assigning a width to the template.

Implementation:

```text
psrism/intrinsic_width.py
psrism/fit_tau.py
psrism/anisotropic_scattering.py
psrism/cli.py
```

### Dispersion-Measure Refinement

#### Per-observation `pdmp` measurement

The reference observing workflow determines the actual DM of each observation with PSRCHIVE `pdmp`, which searches for the DM that maximizes the integrated-profile signal-to-noise ratio, and uses that value in subsequent processing. [2, PDF p. 3, §2.1; 3, PDF p. 68, §4.6]

Run the same optional step for one archive:

```bash
psrism ARCHIVE --measure-dm
```

PSRISM runs `pdmp` on the original archive before its own dedispersion or scrunching, parses `Best DM`, `Correction`, `Error`, and the available best-S/N and period fields, and then dedisperses the in-memory archive at the measured DM. It does not modify the source archive. The `pdmp`-reported DM error is retained as supplied rather than replaced by a PSRISM uncertainty estimate.

`pdmp` is part of PSRCHIVE and normally also requires a working TEMPO installation. Activate the same environment used for PSRCHIVE and verify both dependencies before a run:

```bash
which pdmp
echo "$TEMPO"
pdmp -h
```

If TEMPO is installed but its directory is not exported, set it for the shell or pass it only to PSRISM:

```bash
export TEMPO=/path/to/tempo
psrism ARCHIVE --measure-dm

psrism ARCHIVE --measure-dm --pdmp-tempo-path /path/to/tempo
```

An explicit executable and search controls can be supplied as follows:

```bash
psrism ARCHIVE --measure-dm \
  --pdmp-executable /path/to/pdmp \
  --pdmp-dm-range 0.5 \
  --pdmp-dm-step 0.01 \
  --pdmp-max-channels 128 \
  --pdmp-max-subints 64 \
  --pdmp-max-bins 512 \
  --pdmp-timeout 600
```

The controls map to `pdmp -dr`, `-ds`, `-mc`, `-ms`, and `-mb`, respectively. `--pdmp-dm-range` is a positive half-range around the archive DM, not a full range. Leaving a search or scrunch control unset lets `pdmp` choose its own default. The 300-second default timeout is a PSRISM operational choice and can be changed with `--pdmp-timeout`.

PSRISM invokes `pdmp -f -g /null` in a temporary working directory because `pdmp` can create auxiliary files. It requires a parsed `Best DM` result even when the process exits with status zero, preventing setup errors such as a missing TEMPO configuration from being mistaken for measurements. These execution and validation rules are PSRISM integration choices.

`--measure-dm` and the manual `--dm VALUE` override are mutually exclusive. Use `--dm` when a known finite, nonnegative DM should be imposed without measurement; use `--measure-dm` when this observation should be measured independently.

For a single archive, a successful measurement prints the archive DM, measured DM and uncertainty, correction, and best profile S/N. It also saves:

```text
*_dm_measurement.json
```

The JSON record contains the input archive path and DM, parsed values, executable path, and exact argument list. Raw `pdmp` console text is deliberately omitted to keep the audit file compact.

In directory mode, `--measure-dm` runs independently for every epoch. The time-series CSV contains `dm`, `dm_error`, `dm_method`, `archive_dm`, `dm_correction`, and `pdmp_snr`; `dm_method` is `pdmp_snr` for a measurement, `manual_override` for `--dm VALUE`, and `archive_metadata` otherwise. A `pdmp` failure skips that epoch and records the error in `*_time_series_terminal_output.txt` rather than silently inserting the archive metadata value.

#### Scattering-aware residual-DM check

Cold-plasma dispersion produces a relative arrival-time delay described by the Handbook relation below, with frequency in MHz, DM in pc cm^-3, and delay in seconds. [1, §4.1.1, Eqs. 4.4-4.7]

```math
\Delta t_i
=
\mathcal{D}\,\Delta{\rm DM}
\left(
\nu_i^{-2}-\nu_{\rm ref}^{-2}
\right),
\qquad
\mathcal{D}=4.148808\times10^3
\ {\rm MHz^2\,pc^{-1}\,cm^3\,s}.
```

Maximizing the S/N of a scattered profile can overestimate DM because frequency-dependent scattering shifts the observed peak; aligning the deconvolved intrinsic components instead provides a scattering-aware correction. [6, PDF p. 3, §3]

For a single archive, request the diagnostic together with the isotropic subband fits:

```bash
psrism ARCHIVE --fit-alpha --check-scattering-dm --tau-subbands 6
```

For every accepted alpha subband, PSRISM takes the fitted unscattered phase: the Gaussian centroid for the default one-component model, or the fitted phase shift for an intrinsic template. It periodically unwraps those phases in frequency order and performs a weighted linear fit:

```math
t_i
=
t_0
+
\mathcal{D}\,\Delta{\rm DM}
\left(
\nu_i^{-2}-\nu_{\rm ref}^{-2}
\right),
\qquad
{m DM}_{\rm corrected}
=
{m DM}_{\rm input}+\Delta{\rm DM}.
```

The dispersion relation and sign follow the cited cold-plasma delay law. [1, §4.1.1, Eqs. 4.4-4.7] Weighting by fitted phase covariance, choosing the alpha-fit reference frequency, periodic unwrapping, and requiring at least three accepted subbands are PSRISM implementation choices. If any retained phase lacks a positive covariance error, the fit falls back to unweighted least squares and estimates its covariance from the residual scatter.

With `--profile-components N` for `N > 1`, PSRISM uses the highest-amplitude fitted Gaussian as the phase fiducial in each subband. This component-selection rule is a PSRISM choice and can become ambiguous when component amplitudes cross with frequency; a common `--intrinsic-template` or the default one-component model is safer for this diagnostic.

The result reports `delta_dm`, its covariance uncertainty, corrected DM, reference frequency, reduced chi-square, RMS timing residual, and number of accepted phases in `*_dm_measurement.json`. When `--measure-dm` is also present, the measured `pdmp` DM is the input DM and both results occupy the same JSON report:

```bash
psrism ARCHIVE --measure-dm --fit-alpha --check-scattering-dm
```

Directory mode automatically performs the needed alpha subband fits when the check is requested, even if `alpha` is not selected as a plotted time-series parameter:

```bash
psrism ARCHIVE_DIR --measure-dm --check-scattering-dm \
  --time-params dm,alpha --tau-subbands 6
```

Its CSV adds `scattering_delta_dm`, `scattering_delta_dm_error`, `scattering_corrected_dm`, and `scattering_dm_reduced_chi_square`. The corrected value is a diagnostic and does not trigger a second preprocessing pass. Phase wrapping, intrinsic profile evolution, subband fit degeneracy, and component misidentification can all bias it, so inspect the subband-fit plot and goodness statistics before interpreting a small correction physically.

Implementation:

```text
psrism/dm_analysis.py
psrism/fit_tau.py
psrism/cli.py
psrism/time_series_analysis.py
```

### Anisotropic Pulse Broadening

The scalar exponential broadening model assumes that the scattering angle distribution is isotropic. An asymmetric Gaussian distribution of scattering angles introduces two characteristic scattering times along the screen axes. [4, PDF p. 4, §2.2.2] PSRISM implements the corresponding anisotropic pulse-broadening function with the same periodic train+DC and intrinsic-profile choices as the isotropic fitter: [4, PDF p. 4, Eq. 4]

```math
f_t(t)=
\frac{1}{\sqrt{\tau_x\tau_y}}
\exp\left[
-
\frac{t}{2}
\left(
\frac{1}{\tau_x}
+
\frac{1}{\tau_y}
\right)
\right]
I_0
\left[
\frac{t}{2}
\left(
\frac{1}{\tau_x}
-
\frac{1}{\tau_y}
\right)
\right]
U(t).
```

Here `I_0` is the modified Bessel function of the first kind. [4, PDF p. 4, Eq. 4] PSRISM writes causality explicitly with the unit step `U(t)`. The screen-axis scattering times are: [4, PDF p. 4, immediately below Eq. 4]

```math
\tau_x =
\frac{D_s'\sigma_{a,x}^2}{c},
\qquad
\tau_y =
\frac{D_s'\sigma_{a,y}^2}{c}.
```

The scalar summary used for frequency scaling is the geometric mean discussed for the anisotropic simulations: [4, PDF p. 10, Fig. 14 discussion]

```math
\tau_{\rm eff}
=
\sqrt{\tau_x\tau_y}.
```

The fitted `tau_ratio` below is a PSRISM parameterization of the two reference timescales, not a separately defined literature observable:

```math
{\rm tau\_ratio}
=
\frac{
\max(\tau_x,\tau_y)
}{
\min(\tau_x,\tau_y)
}.
```

This is not the same as the annual-scintillation axial ratio `A_r` used in multi-epoch velocity modelling. `A_r` requires observations across many epochs, projected Earth velocity, pulsar proper motion, and a screen-velocity model. The current CLI implements the single-archive profile-broadening anisotropy test.

In the code, the free parameters are `A`, `μ`, `σ`, `τ_eff`, `tau_ratio`, and a constant baseline `C`. The fitted baseline and pulse-train treatment follow the train+DC motivation. [3, PDF p. 33, §2.7.2; 4, PDF pp. 7-8, §2.4] The following conversion is PSRISM's parameterization of the geometric mean and ratio:

```math
\tau_x =
\frac{\tau_{\rm eff}}{\sqrt{{\rm tau\_ratio}}},
\qquad
\tau_y =
\tau_{\rm eff}\sqrt{{\rm tau\_ratio}}.
```

For folded pulsar profiles, scattering power can wrap into the next rotation and raise the apparent baseline. [3, PDF p. 33, §2.7.2; 4, PDF pp. 7-8, §2.4] PSRISM handles this by folding the anisotropic pulse-broadening function over several pulse periods and using circular convolution with a one-Gaussian, multi-Gaussian, or template intrinsic profile. The same `--profile-components` and `--intrinsic-template` options apply to `--fit-anisotropy`.

Implementation:

```text
psrism/anisotropic_scattering.py
psrism/fit_alpha.py
psrism/plot_tau_vs_freq.py
```

Relevant functions:

```text
anisotropic_pbf
anisotropic_scattered_pulse
fit_anisotropic_profile
fit_anisotropic_scattering_from_archive
plot_anisotropic_subband_fits
```

Run:

```bash
psrism ARCHIVE --fit-anisotropy --tau-subbands 4
```

This command reports `τ_x`, `τ_y`, `τ_eff`, `tau_ratio`, fit errors, and fit quality for each subband. The `τ_eff` frequency-scaling path uses the same effective monochromatic frequencies, subband acceptance rules, minimum accepted count, Monte Carlo intervals, conservative errors, CLI controls, and accepted/rejected plotting styles documented for the isotropic α fit above. It saves:

```text
*_anisotropic_tau_eff_vs_freq.png
*_anisotropic_subband_fits.png
```

The `*_anisotropic_tau_eff_vs_freq.png` caption reports the same weighted power-law goodness and Monte Carlo values for the `τ_eff(ν)` fit.

Terminal output includes:

```text
Anisotropic scattering fits:
 channels 0-90: fc=... MHz, bandwidth=... MHz, fm=... MHz, tau_x=... s, tau_y=... s, tau_eff=... s, tau_ratio=...
  goodness: anisotropic red. chi-square = ..., isotropic red. chi-square = ..., RMS = ..., fit S/N = ..., valid channels = ..., status = accepted

anisotropic tau_eff alpha = ... +.../-... (adopted +/- ...)
alpha convention: tau(nu) = tau_ref * (nu / nu_ref)^(-alpha)
tau_eff,150 = ... +.../-... s (adopted +/- ... s) at nu_ref=150 MHz
```

The isotropic reduced chi-square is printed as a diagnostic comparison. It is useful for checking whether the anisotropic fit reduces residual structure, but it is not by itself proof of anisotropy; profile evolution, baseline errors, finite screens, and RFI can also bias scattering fits.

### Annual Scintillation Anisotropy

The annual anisotropy model describes scintillation velocities measured over many epochs. It is distinct from the single-archive pulse-broadening anisotropy fit above and combines `Delta_nu_d`, `Delta_t_d`, projected Earth velocity, pulsar proper-motion velocity, distance, and a screen-velocity model. [7, PDF pp. 2-3, §§2-3]

Run it on an archive directory, supplying an independently sourced pulsar distance and equatorial proper motion:

```bash
psrism ARCHIVE_DIR --fit-annual-anisotropy \
  --annual-distance-kpc DISTANCE \
  --annual-pm-ra-cosdec-mas-yr PM_RA_COS_DEC \
  --annual-pm-dec-mas-yr PM_DEC
```

`--annual-pm-ra-cosdec-mas-yr` explicitly expects `mu_RA cos(dec)`, not unscaled `mu_RA`. PSRISM converts both proper-motion components to km/s using the standard `4.74047 mu D` astrometric conversion with proper motion in mas/yr and distance in kpc. The distance and proper-motion values are user inputs: PSRISM does not silently query or adopt a catalogue value.

The option is directory-only and automatically adds `dnu_d` and `dt_d` to the requested time-series parameters. Only resolved, finite ACF measurements with a positive `Delta_t_d` uncertainty, observing frequency, epoch, and source coordinates enter the fit. The per-epoch centre frequency is retained as `observing_frequency_mhz` in the CSV.

The observed scintillation velocity is: [7, PDF p. 3, Eq. 1]

```math
V_{\rm ISS}
=
A_{\rm ISS}
\frac{\sqrt{D\Delta\nu_{\rm d}}}{f\Delta t_{\rm d}}.
```

Here `D` is in kpc, `Delta_nu_d` in MHz, `f` in GHz, `Delta_t_d` in seconds, and velocity in km/s. For an anisotropic thin screen PSRISM uses: [7, PDF p. 3, §3.1, immediately below Eq. 1]

```math
A_{\rm ISS}
=
2.78\times10^4
\sqrt{\frac{A_r+1/A_r}{2}}
\sqrt{\frac{2D_s}{D-D_s}}.
```

The observed velocity and thin-screen prediction are related by: [7, PDF p. 3, Eq. 2]

```math
V_{\rm ISS}
=
|V_{\rm eff}|\frac{D}{D-D_s}.
```

For the currently supported solitary-pulsar model, the effective equatorial components are: [7, PDF p. 3, Eqs. 3 and 5]

```math
V_{{\rm eff},\alpha}
=
\frac{D-D_s}{D}V_{E,\alpha}
+
\frac{D_s}{D}V_{\mu,\alpha}
-
V_{{\rm IISM},\alpha},
```

```math
V_{{\rm eff},\delta}
=
\frac{D-D_s}{D}V_{E,\delta}
+
\frac{D_s}{D}V_{\mu,\delta}
-
V_{{\rm IISM},\delta}.
```

The reference equation also includes binary orbital transverse velocity. [7, PDF p. 3, Eq. 3] PSRISM does not yet evaluate a binary orbit, so this CLI fit must not be used for a binary pulsar unless its orbital contribution is demonstrably negligible.

For an anisotropic scintillation pattern: [7, PDF p. 3, Eq. 5]

```math
|V_{\rm eff}|
=
\sqrt{
aV_{{\rm eff},\alpha}^2
+
bV_{{\rm eff},\delta}^2
+
cV_{{\rm eff},\alpha}V_{{\rm eff},\delta}
}.
```

The coefficients and orientation convention are: [7, PDF p. 3, Eq. 6]

```math
R = \frac{A_r^2-1}{A_r^2+1},
\qquad
a = \frac{1-R\cos(2\psi)}{\sqrt{1-R^2}},
\qquad
b = \frac{1+R\cos(2\psi)}{\sqrt{1-R^2}},
\qquad
c = \frac{-2R\sin(2\psi)}{\sqrt{1-R^2}}.
```

`A_r >= 1` is the axial ratio, and `psi` is measured clockwise from the pulsar's right-ascension direction. [7, PDF p. 3, Eq. 6] PSRISM obtains Earth's built-in barycentric velocity from Astropy and projects it onto the ICRS RA/Dec tangent basis. The reference implementation obtains Earth velocity with SCINTOOLS. [7, PDF pp. 3-4, §3.2] Astropy projection is a PSRISM implementation choice, and station-rotation velocity is omitted.

The fit uses the combined observable: [7, PDF p. 3, Eq. 4]

```math
Q
=
\frac{\sqrt{\Delta\nu_{\rm d}}}{f\Delta t_{\rm d}}
=
\frac{|V_{\rm eff}|\sqrt{D}}{A_{\rm ISS}(D-D_s)}.
```

Following the reference analysis, PSRISM substitutes the arithmetic mean of all usable `Delta_nu_d` measurements for each epoch's bandwidth. [7, PDF p. 3, §3.2] If every bandwidth has a positive uncertainty, the mean uncertainty is propagated assuming independent measurements; otherwise PSRISM uses the empirical standard error of the measured bandwidths. It then applies first-order independent propagation:

```math
\frac{\sigma_Q}{Q}
=
\sqrt{
\left(\frac{\sigma_{\overline{\Delta\nu}_{\rm d}}}{2\overline{\Delta\nu}_{\rm d}}\right)^2
+
\left(\frac{\sigma_{\Delta t_{\rm d}}}{\Delta t_{\rm d}}\right)^2
}.
```

The propagation formula and the treatment of the shared mean-bandwidth error as independent between epoch likelihood terms are PSRISM choices. Frequency, distance, and proper-motion uncertainties are not marginalized, so their effect must be assessed separately when interpreting the posterior.

The five fitted parameters are `V_IISM,alpha`, `V_IISM,delta`, `A_r`, `psi`, and `D_s`, matching the reference fit. [7, PDF p. 4, §3.2] A deterministic multi-start bounded least-squares pass initializes the walkers; final parameter intervals come from the `emcee` ensemble MCMC with uniform priors, as in the reference method. [7, PDF p. 4, §3.2] MCMC is statistical posterior sampling, not machine-learning analysis. The posterior median is reported as the point estimate, with 16th-to-50th and 50th-to-84th percentile errors. The bundled paper displays the same outer percentiles around its selected central estimate. [7, PDF p. 7, Fig. A.1]

Generic prior bounds and sampling defaults are PSRISM choices:

- Each IISM velocity component is uniform from `-500` to `+500 km/s`; change the symmetric component bound with `--annual-max-screen-speed`.
- `A_r` is uniform from `1` to `20`; change the upper bound with `--annual-max-axial-ratio`.
- `psi` is uniform from `0` to `180 degrees`.
- `D_s` is uniform inside the line of sight, from `1e-4 D` to `(1-1e-4)D`.
- At least eight usable epochs are required by default; change this with `--annual-min-epochs`, whose mathematical minimum is six for five fitted parameters.
- Sampling defaults to 32 walkers, 5000 steps, 1000 discarded burn-in steps, and seed zero. Configure these with `--annual-mcmc-walkers`, `--annual-mcmc-steps`, `--annual-mcmc-burn`, and `--annual-mcmc-seed`.

Passing the minimum epoch count does not guarantee useful annual phase coverage. The JSON therefore retains each epoch's day of year and the complete MJD span; inspect their distribution before treating the fitted screen parameters as physical measurements.

For example, a longer run with narrower generic bounds is:

```bash
psrism ARCHIVE_DIR --fit-annual-anisotropy \
  --annual-distance-kpc DISTANCE \
  --annual-pm-ra-cosdec-mas-yr PM_RA_COS_DEC \
  --annual-pm-dec-mas-yr PM_DEC \
  --annual-max-screen-speed 200 --annual-max-axial-ratio 10 \
  --annual-mcmc-walkers 64 --annual-mcmc-steps 20000 \
  --annual-mcmc-burn 5000 --annual-mcmc-seed 17
```

The generated products are:

```text
*_time_series_annual_anisotropy.json
*_time_series_annual_anisotropy_chain.npz
*_time_series_annual_anisotropy.png
*_time_series_annual_anisotropy_posterior.png
```

The JSON contains inputs, bounds, MCMC configuration, mean acceptance fraction, estimated autocorrelation times, chi-square, reduced chi-square, posterior summaries, and per-epoch Earth velocities, `Q`, model values, scintillation velocities, and normalized residuals. The compressed NPZ retains every finite post-burn sample, its log probability, and column names. The annual plot shows observed and model scintillation velocity against day of year with normalized residuals; the posterior plot shows one- and two-dimensional parameter distributions. A chain much shorter than the reported autocorrelation times or a poor acceptance fraction is not a converged physical measurement and should be rerun or reconsidered.

Implementation:

```text
psrism/annual_anisotropy.py
psrism/time_series_analysis.py
psrism/cli.py
```

Relevant functions:

```text
anisotropy_coefficients
effective_velocity_components
anisotropic_effective_speed
thin_screen_a_iss
annual_q_model
earth_velocity_equatorial_kms
fit_annual_anisotropy
```

### Refractive Scintillation Estimates

`psrism` can derive scintillation quantities from a fitted scattering time. For each fitted subband, the decorrelation bandwidth implied by the scattering time is: [1, §4.2.5.2, Eq. 4.39; 5, PDF p. 6, Eq. 8]

```math
\Delta\nu_{\rm d}
=
\frac{C_1}{2\pi\tau}.
```

For Kolmogorov turbulence, the adopted default is: [1, §4.2.3; 5, PDF p. 6, Eq. 8]

```math
C_1 = 1.16.
```

If the pulsar distance `D` and effective transverse speed `V` are supplied, the diffractive scintillation timescale is estimated by rearranging the standard scintillation-velocity equation with `A = 2.53 x 10^4` in the stated units: [1, §7.4.4.1, Eq. 7.39]

```math
\Delta t_{\rm d}
=
\frac{
2.53\times10^4
\sqrt{D\Delta\nu_{\rm d}}
}{
\nu_{\rm GHz}V
}.
```

Here `D` is in kpc, `Δν_d` is in MHz, `ν_GHz` is the observing frequency in GHz, `V` is in km/s, and `Δt_d` is in seconds.

The finite-scintle uncertainty is estimated using: [1, §4.2.5.2; 5, PDF p. 5, Eqs. 6-7]

```math
N_{\rm scintles}
=
\left(
1+
\eta_t
\frac{T}{\Delta t_{\rm d}}
\right)
\left(
1+
\eta_\nu
\frac{B}{\Delta\nu_{\rm d}}
\right),
```

The filling factors are empirical and commonly lie between 0.1 and 0.2; PSRISM adopts the memo's values: [1, §4.2.5.2; 5, PDF p. 5, Eq. 7]

```math
\eta_t = 0.2,
\qquad
\eta_\nu = 0.2.
```

The finite-scintle contribution to the τ uncertainty is: [5, PDF p. 5, Eq. 6]

```math
\sigma_{\rm fse}
=
\frac{\tau}{\sqrt{N_{\rm scintles}}},
```

PSRISM combines the fitted and finite-scintle terms in quadrature as an independent-error assumption:

```math
\sigma_{\rm total}
=
\sqrt{
\sigma_{\rm fit}^2
+
\sigma_{\rm fse}^2
}.
```

The refractive scintillation timescale is: [5, PDF p. 9, Eq. 14]

```math
T_{\rm r}
=
\frac{4}{\pi}
\frac{\nu}{\Delta\nu_{\rm d}}
\Delta t_{\rm d}.
```

Use consistent units for `ν` and `Δν_d`; in the code both are in MHz, so `T_r` has the same time unit as `Δt_d`.

Implementation:

```text
psrism/refractive_scintillation.py
```

Relevant function:

```text
estimate_scintillation_from_tau
finite_scintle_statistics
```

Run with isotropic subband τ fits:

```bash
psrism ARCHIVE --estimate-refractive --tau-subbands 4
```

Override the literature-based defaults explicitly with `--c1`, `--eta-time`, and `--eta-freq`:

```bash
psrism ARCHIVE --estimate-refractive \
  --c1 1.16 \
  --eta-time 0.2 \
  --eta-freq 0.2
```

The two `eta` options also control the finite-coverage diagnostic written by direct `--acspec` and directory ACF measurements. `--c1` applies only when converting a fitted scattering time into a decorrelation bandwidth.

This reports `Δν_d` from τ. It uses the same stage-4 subband quality controls as `--fit-alpha` and sends only accepted subbands into the derived scintillation calculation. Add distance and velocity together to estimate `Δt_d`, `N_scintles`, `σ_fse`, `σ_total`, and `T_r`; supplying only one is rejected because the velocity relation requires both:

```bash
psrism ARCHIVE --estimate-refractive --tau-subbands 4 \
  --distance-kpc 1.0 \
  --velocity-kms 100
```

Use anisotropic `τ_eff` instead of isotropic τ by combining:

```bash
psrism ARCHIVE --fit-anisotropy --estimate-refractive --tau-subbands 4 \
  --distance-kpc 1.0 \
  --velocity-kms 100
```

Terminal output includes:

```text
Scintillation and refractive estimates from anisotropic tau_eff:
 nu=... MHz, tau=... s, Delta_nu_d=... MHz
  Delta_t_d=... s, T_r=... days, N_scintles=..., sigma_fse=... s, sigma_total=... s
```

## Module Map

```text
psrism/cli.py                              Command-line interface
psrism/archive_io.py                       Archive loading, preprocessing, and NumPy conversion
psrism/data_quality.py                     Dynamic-spectrum quality assessment and RFI masks
psrism/dynamic_flux.py                     Profile-window detection and dynamic-spectrum flux extraction
psrism/dynamic_spectrum.py                 Dynamic-spectrum construction and normalization
psrism/plot_dynamic_spectrum.py            Dynamic-spectrum and integrated-profile plotting
psrism/autocorrelation_spectrum.py         Autocorrelation-spectrum calculation
psrism/plot_autocorrelation_spectrum.py    Autocorrelation-spectrum plotting
psrism/scintillation_spectrum.py           Secondary-spectrum calculation
psrism/plot_scintillation_spectrum.py      Secondary-spectrum plotting
psrism/fit_secondary_spectrum.py           Parabolic-arc fitting
psrism/anisotropic_scattering.py           Anisotropic pulse-broadening fitting
psrism/annual_anisotropy.py                Annual anisotropy MCMC fitting and diagnostics
psrism/batch_analysis.py                   Mixed-source grouping and cross-pulsar catalogue exports
psrism/refractive_scintillation.py         Derived scintillation and refractive estimates
psrism/profile_models.py                   Shared periodic intrinsic profiles and broadening kernels
psrism/intrinsic_width.py                  Intrinsic Gaussian FWHM and frequency-evolution diagnostics
psrism/fit_tau.py                          Scattered-pulse model and τ fitting
psrism/fit_alpha.py                        Frequency-scaling fit for α
psrism/plot_tau_vs_freq.py                 τ-versus-frequency plotting
psrism/dm_analysis.py                      pdmp integration and scattering-aware DM checks
psrism/time_series_analysis.py             Directory time-series CSV writing and plotting
psrism/temporal_statistics.py              Non-ML temporal tests and correlation analysis
psrism/dm_variability.py                    DM slopes and solar-separation diagnostics
psrism/solar_geometry.py                    Observation timing and Sun-pulsar geometry
psrism/bandpass.py                         Dynamic-spectrum bandpass scaling
psrism/dm_vs_time.py                       DM-versus-time helpers and plotting
psrism/tau_vs_time.py                      τ-versus-time helpers and plotting
psrism/fit_autocorrelation_spectrum.py     Autocorrelation-spectrum fitting
```

## References

Citation locations refer to the PDF page shown by a PDF reader, not the printed journal or thesis page, unless a Handbook section is given. `psrhandbook_chapter4.txt` is an OCR aid derived from reference [1], not a separate scientific source.

[1] D. R. Lorimer and M. Kramer, *Handbook of Pulsar Astronomy*, Cambridge University Press (2005). Bundled file: `psrhandbook.pdf`. Citations use the lowest applicable section number, as requested.

[2] A. Filothodoros et al., “Observations of interstellar scattering of six pulsars using Polish LOFAR station PL611,” *Monthly Notices of the Royal Astronomical Society* **528**, 5667-5678 (2024), doi:10.1093/mnras/stae399. Bundled file: `Alex_and_Boe.pdf`.

[3] A. Filothodoros, doctoral thesis on pulsar scattering observations and temporal variability. Bundled file: `ALEX.pdf`. The supplied PDF begins with the contents pages and does not include a machine-readable title page, so citations identify both PDF page and thesis section.

[4] M. Geyer and A. Karastergiou, “The frequency dependence of scattering imprints on pulsar observations,” *Monthly Notices of the Royal Astronomical Society* **462**, 2587-2602 (2016), doi:10.1093/mnras/stw1724. Bundled file: `fitting.pdf`.

[5] A. Geiger and M. Lam, “The Frequency-Dependent Scattering of Pulsar J1903+0327,” NANOGrav Memorandum 008 (2022). Bundled file: `NANOGrav-Memo-008.pdf`.

[6] M. Geyer and A. Karastergiou, “Anomalous Pulsar Scattering at LOFAR Frequencies,” *Proceedings of IAU Symposium 337* (2018), doi:10.1017/S1743921317008407. Bundled file: `anomalous-pulsar-scattering-at-lofar-frequencies.pdf`.

[7] Y. Cai et al., “Pulsar scintillation studies with LOFAR III. Annual variations in PSR J0814+7429,” arXiv:2604.02681v1 (2026). Bundled file: `scintillationJ0814.pdf`.

## Quick Fixes

### Matplotlib cache warning

The CLI sets a writable default matplotlib cache directory in `/tmp`. If matplotlib still reports that `~/.config/matplotlib` is not writable, set a cache directory explicitly before running plots:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-$USER
```

### Command not found

If `psrism` is not found after cloning the repository, install the package in the active environment:

```bash
cd psrism
python -m pip install -e .
```

Then verify:

```bash
which psrism
psrism --help
```

### Invalid scrunch target

If you see an error such as:

```text
--nchan=128 is not valid for current nchan=366
```

inspect the archive:

```bash
psrism ARCHIVE --inspect
```

Choose `--nsub`, `--nchan`, and `--nbin` values from the valid target lists printed by the command.
