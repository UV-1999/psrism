# PSRISM

`psrism` is a command-line Python package for pulsar interstellar-medium analysis from PSRCHIVE-readable archive files. It is designed for studying pulse broadening, diffractive scintillation, scintillation bandwidths and timescales, secondary spectra, and the frequency scaling of scattering in radio pulsar observations.

Scientifically, `psrism` turns a calibrated pulsar archive into analysis products such as dynamic spectra, autocorrelation spectra, secondary spectra, integrated pulse profiles, scattering timescales τ, and scattering spectral indices α. It uses `psrchive` for archive I/O and standard scientific Python tools for numerical work, fitting, and plotting. Python dependencies are listed in [`requirements.txt`](requirements.txt). `psrchive` is usually installed separately, for example through conda-forge or a local PSRCHIVE build.

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
  ARCHIVE_STEM_dspec.png
  ARCHIVE_STEM_acspec.png
  ARCHIVE_STEM_sspec.png
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
psrism /home/piyushmarmat/old/Data/J0139+5814 \
  --time-params all \
  --nsub 70 \
  --nchan 61 \
  --nbin 256 \
  --tau-subbands 4
```

Run a lighter time series with only archive metadata and profile-fitting parameters:

```bash
psrism /home/piyushmarmat/old/Data/J0139+5814 \
  --time-params dm,tau,alpha
```

Restrict the run to one archive or one subset of archives:

```bash
psrism /home/piyushmarmat/old/Data/J0139+5814 \
  --time-pattern "J0139+5814_2019-08*.nop" \
  --time-params dnu_d,dt_d,t_r \
  --nsub 70 \
  --nchan 61 \
  --nbin 256
```

The supported parameter names are:

```text
dm       dispersion measure from the archive metadata
tau      integrated-profile scattering timescale
alpha    scattering spectral index from subband tau fits
dnu_d    ACF decorrelation bandwidth
dt_d     ACF diffractive scintillation timescale
t_r      refractive scintillation timescale inferred from ACF dnu_d and dt_d
all      all of the above
```

Epochs are read from the PSRCHIVE integration epoch when available. If that fails, `psrism` falls back to the archive filename pattern `YYYY-MM-DD_HH:MM:SS`. The MJD conversion uses:

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

For each archive, the directory workflow saves one CSV table and one plot for each requested parameter:

```text
J0139+5814/
  J0139+5814_time_series_terminal_output.txt
  J0139+5814_time_series.csv
  J0139+5814_time_series_dm_vs_time.png
  J0139+5814_time_series_tau_vs_time.png
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

The refractive timescale plotted as `t_r` is computed from those measured ACF quantities:

```math
T_{\rm r}
=
\frac{4}{\pi}
\frac{\nu}{\Delta\nu_{\rm d}}
\Delta t_{\rm d}.
```

Here `ν` and `Δν_d` are both in MHz, so `T_r` has the same time unit as `Δt_d`; the plot reports it in days.

Implementation:

```text
psrism/cli.py
psrism/time_series_analysis.py
psrism/autocorrelation_spectrum.py
psrism/dynamic_spectrum.py
psrism/fit_tau.py
psrism/fit_alpha.py
```

## Plotting

All heatmap-style plots use the `afmhot` color map. The plot title is the archive file name.

### Dynamic Spectrum

The dynamic spectrum shows pulse intensity as a function of observing time and radio frequency. It is the primary input for scintillation analysis.

For each subintegration and frequency channel, `psrism` estimates an on-pulse flux from the noise-normalized pulse profile:

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

The output file is:

```text
*_dspec.png
```

### Autocorrelation Spectrum

The autocorrelation spectrum measures how similar the dynamic spectrum is to itself after a time lag and a frequency lag. It is used to estimate the characteristic diffractive scintillation bandwidth and timescale.

Following the Cordes/Lorimer-Kramer convention, the finite-lag covariance is:

```math
{\rm CF}(\Delta\nu,\Delta t)=
\sum_{\nu=1}^{n_\nu-|\Delta\nu|}
\sum_{t=1}^{n_t-|\Delta t|}
E(\nu,t)\,
E(\nu+\Delta\nu,t+\Delta t).
```

The normalized autocorrelation function is:

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

When `--acspec` is used, `psrism` also reports two quantities to the terminal.

The decorrelation bandwidth is measured from the zero-time-lag frequency cut:

```math
\Delta\nu_{\rm DISS}:
{\rm ACF}(\Delta\nu,0)=\frac{1}{2}.
```

The diffractive scintillation timescale is measured from the zero-frequency-lag time cut:

```math
\Delta t_{\rm DISS}:
{\rm ACF}(0,\Delta t)=\frac{1}{e}.
```

This measurement is implemented in `measure_acf_scales` in [`psrism/autocorrelation_spectrum.py`](psrism/autocorrelation_spectrum.py). Terminal output is printed by the CLI in [`psrism/cli.py`](psrism/cli.py), for example:

```text
ACF decorrelation measurements:
 decorrelation bandwidth = 0.42 MHz
 diffractive timescale = 185 s
```

#### Tilted ACF and Scintle Drift

In an ideal stationary diffractive scintillation pattern, the central ACF lobe is roughly elliptical and aligned with the time-lag and frequency-lag axes. A tilted central ellipse means that scintillation maxima at one frequency are correlated with maxima at another frequency after a shifted time lag. This is usually interpreted as frequency drift of scintles caused by large-scale refractive phase gradients in the ionized interstellar medium.

The physical picture is that a refractive phase gradient displaces the apparent scattered image. Because the displacement is chromatic, the diffraction pattern shifts with observing frequency. If this shift has a component parallel to the effective transverse velocity, the dynamic spectrum shows tilted scintillation bands and the ACF becomes skewed. This interpretation is discussed in classic scintillation-drift work such as Smith & Wright (1985) and Gupta, Rickett & Lyne (1994).

When `--acspec` is used, `psrism` fits the central ACF lobe with a tilted elliptical Gaussian:

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

The fitted ridge of maximum correlation satisfies:

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

The terminal output includes:

```text
Tilted ACF Gaussian fit:
 fit decorrelation bandwidth = ... MHz
 fit diffractive timescale = ... s
 correlation coefficient = ...
 ellipse rotation angle = ... deg
 scintle drift slope d(time lag)/d(freq lag) = ... s/MHz
 inverse drift rate d(freq lag)/d(time lag) = ... MHz/s
 quadratic coefficients: a=..., b=..., c=...
 goodness: unweighted reduced chi-square = ..., RMS residual = ..., fit points = ...
```

### Secondary Spectrum

The secondary spectrum is the squared magnitude of the two-dimensional Fourier transform of the mean-subtracted dynamic spectrum. It represents scintillation power in conjugate time-frequency space.

First the dynamic spectrum is mean-subtracted:

```math
E'(t,\nu)=E(t,\nu)-\langle E\rangle.
```

Then:

```math
S(f_t,f_\nu)=
\left|
{\cal F}_2\left\{E'(t,\nu)\right\}
\right|^2.
```

For plotting, `psrism` uses:

```math
S_{\rm dB}=10\log_{10}\left(S+10^{-12}\right).
```

The FFT-shifted axes are made with `numpy.fft.fftfreq`. The x-axis is fringe frequency in Hz. The y-axis is delay in seconds because the frequency-channel spacing is converted from MHz to Hz before building the Fourier-conjugate axis. The zero fringe-frequency row is set to zero before display and arc fitting.

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

#### Parabolic Arc Fitting

Secondary spectra often show parabolic scintillation arcs because the dynamic spectrum is produced by interference between scattered paths. For a thin scattering screen, a path scattered by angle `θ` has a differential geometric delay that scales approximately as:

```math
\tau_{\rm d}
\propto
\theta^2,
```

while the Doppler or fringe frequency scales approximately as:

```math
f_{\rm D}
\propto
\theta.
```

Eliminating `θ` gives the parabolic arc relation:

```math
\tau_{\rm d}
=
\eta f_{\rm D}^2.
```

The curvature `η` is the fitted observable. In a simple thin-screen geometry it is related to effective distance, observing wavelength, and effective transverse velocity by:

```math
\eta
=
\frac{
D_{\rm eff}\lambda^2
}{
2cV_{\rm eff,\parallel}^2
}.
```

This relation is physically useful, but it is model-dependent: anisotropy, multiple screens, screen motion, or extended scattering material can change the interpretation of `η`. Parabolic arcs and arclets are discussed in Stinebring et al. (2001), Walker et al. (2004), and Cordes et al. (2006).

In `psrism`, arc fitting is requested with:

```bash
psrism ARCHIVE --sspec --fit-arc
```

The fitted model is:

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

By default, the apex offsets are zero:

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

The code uses an arc-strength search, similar in spirit to a Hough transform. For each trial curvature, it samples the linear secondary-spectrum power along the parabola and computes:

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

The best curvature is the trial that maximizes `A(η)`. The uncertainty is estimated from the half-maximum width of the arc-strength curve above its median baseline. The central DC spike and axis leakage are masked before scoring.

Useful options are:

```bash
psrism ARCHIVE --sspec --fit-arc --arc-half positive
psrism ARCHIVE --sspec --fit-arc --arc-half both
psrism ARCHIVE --sspec --fit-arc --arc-curvature-trials 400
psrism ARCHIVE --sspec --fit-arc --arc-curvature-min 1e-3 --arc-curvature-max 1e2
psrism ARCHIVE --sspec --fit-arc --arc-mask-bins 4
```

The implementation is in [`psrism/fit_secondary_spectrum.py`](psrism/fit_secondary_spectrum.py). The secondary-spectrum plot overlays the fitted parabola when `--sspec` and `--fit-arc` are used together.

Terminal output includes:

```text
Parabolic arc fit:
 delay half searched = positive
 apex fringe-frequency offset = 0 Hz
 apex delay offset = 0 s
 curvature eta = ... +/- ... s^3
 arc strength = ...
 arc strength S/N = ...
 samples on best arc = ...
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
 tau (bins) = ...
 tau = ... +/- ... s
 goodness: unweighted reduced chi-square = ..., RMS residual = ...
```

## Fitting

### Scattering Timescale τ

The scattering timescale τ describes the characteristic exponential pulse broadening caused by multipath propagation through the ionized interstellar medium. In `psrism`, the intrinsic pulse is approximated by a Gaussian and convolved with a one-sided exponential pulse broadening function.

The analytic exponentially modified Gaussian model is:

```math
P(t)=
A\frac{\sigma}{\tau}
\sqrt{\frac{\pi}{2}}
\exp\left[-\frac{t-\mu}{\tau}\right]
\left[
1+
{\rm erf}
\left(
\frac{t-\left(\mu+\sigma^2/\tau\right)}
{\sigma\sqrt{2}}
\right)
\right].
```

The fitted parameters are `A`, `μ`, `σ`, and `τ`. The uncertainty on τ is extracted from the covariance matrix returned by nonlinear least squares.

Implementation:

```text
psrism/fit_tau.py
```

The main functions are `scattered_pulse`, `fit_tau_from_profile`, and `fit_tau_from_archive`.

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

### Scattering Spectral Index α

The frequency scaling of scattering is modeled as:

```math
\tau(\nu)=
\tau_0
\left(
\frac{\nu}{\nu_0}
\right)^\alpha.
```

In logarithmic form:

```math
\log_{10}\tau =
\alpha
\log_{10}
\left(
\frac{\nu}{\nu_0}
\right)
+
\log_{10}\tau_0.
```

For a single archive, `psrism` divides the full bandwidth into contiguous frequency subbands. The characteristic frequency of each subband is the central frequency of that subband:

```math
\nu_i =
\frac{\nu_{\rm min,i}+\nu_{\rm max,i}}{2}.
```

Each subband profile is fit independently to obtain:

```math
(\tau_i,\sigma_{\tau_i}).
```

The weighted fit is done in log space:

```math
x_i =
\log_{10}
\left(
\frac{\nu_i}{\nu_0}
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
y_i = \alpha x_i + b,
```

and:

```math
\tau_0 = 10^b.
```

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

Optional reference frequency:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 4 --tau-reference-freq 150
```

Terminal output includes subband τ measurements and the fitted α:

```text
Subband tau fits:
 channels 0-90: nu=... MHz, tau=... +/- ... s
 channels 91-181: nu=... MHz, tau=... +/- ... s

alpha = ... +/- ...
tau0 = ... +/- ... s at nu0=... MHz
```

This command saves:

```text
*_tau_vs_freq.png
*_subband_tau_fits.png
```

The τ-versus-frequency plot also includes a comparison line with fixed `α = -4.4`, normalized to the fitted `τ0` at the same reference frequency. Its global caption reports the power-law fit goodness: weighted χ², reduced χ², degrees of freedom, RMS log residual, and number of fitted points.

The subband fit overlay plot uses an automatic compact grid when no layout is supplied. For prime numbers of subbands, empty panels are hidden. You can choose the grid manually with:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 10 \
  --subband-plot-rows 2 \
  --subband-plot-cols 5
```

If only one grid dimension is supplied, `psrism` computes the other one. The same grid options are used by `--fit-anisotropy`.

### Anisotropic Pulse Broadening

The scalar exponential broadening model assumes that the scattering angle distribution is isotropic. If the scattering screen is elongated, the pulse broadening can be different along two transverse screen axes. `psrism` implements the anisotropic pulse broadening function:

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

Here `I_0` is the modified Bessel function of the first kind and `U(t)` is the unit step function. The screen-axis scattering times are:

```math
\tau_x =
\frac{D_s'\sigma_{a,x}^2}{c},
\qquad
\tau_y =
\frac{D_s'\sigma_{a,y}^2}{c}.
```

The scalar summary used for frequency scaling is:

```math
\tau_{\rm eff}
=
\sqrt{\tau_x\tau_y}.
```

The fitted anisotropy strength reported by the single-archive profile model is:

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

In the code, the free parameters are `A`, `μ`, `σ`, `τ_eff`, `tau_ratio`, and a constant baseline `C`. They are converted internally to:

```math
\tau_x =
\frac{\tau_{\rm eff}}{\sqrt{{\rm tau\_ratio}}},
\qquad
\tau_y =
\tau_{\rm eff}\sqrt{{\rm tau\_ratio}}.
```

For folded pulsar profiles, scattering power can wrap into the next rotation. `psrism` handles this by folding the anisotropic pulse broadening function over several pulse periods and using circular convolution with the intrinsic Gaussian profile.

Implementation:

```text
psrism/anisotropic_scattering.py
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

This command reports `τ_x`, `τ_y`, `τ_eff`, `tau_ratio`, fit errors, and fit quality for each subband. It also fits the frequency scaling of `τ_eff` and saves:

```text
*_anisotropic_tau_eff_vs_freq.png
*_anisotropic_subband_fits.png
```

The `*_anisotropic_tau_eff_vs_freq.png` caption reports the same weighted power-law goodness values for the `τ_eff(ν)` fit.

Terminal output includes:

```text
Anisotropic scattering fits:
 channels 0-90: nu=... MHz, tau_x=... s, tau_y=... s, tau_eff=... s, tau_ratio=...
  goodness: anisotropic red. chi-square = ..., isotropic red. chi-square = ..., RMS = ...

anisotropic tau_eff alpha = ... +/- ...
tau_eff0 = ... +/- ... s at nu0=... MHz
```

The isotropic reduced chi-square is printed as a diagnostic comparison. It is useful for checking whether the anisotropic fit reduces residual structure, but it is not by itself proof of anisotropy; profile evolution, baseline errors, finite screens, and RFI can also bias scattering fits.

### Annual Anisotropy Model Helpers

The attached notes also describe the many-epoch annual anisotropy model for scintillation velocities. This is different from the single-archive pulse-broadening anisotropy fit above. It requires `Δν_d` and `Δt_d` measured at many epochs, plus Earth velocity projected on the sky, pulsar proper-motion velocity, distance, and screen-velocity assumptions.

The observed scintillation velocity is:

```math
V_{\rm ISS}
=
A_{\rm ISS}
\frac{
\sqrt{D\Delta\nu_{\rm d}}
}{
f\Delta t_{\rm d}
}.
```

The thin-screen effective velocity components are:

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

For an anisotropic scintillation pattern:

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

The coefficients are:

```math
R =
\frac{
A_r^2 - 1
}{
A_r^2 + 1
},
```

```math
a =
\frac{1-R\cos(2\psi)}{\sqrt{1-R^2}},
\qquad
b =
\frac{1+R\cos(2\psi)}{\sqrt{1-R^2}},
```

```math
c =
\frac{-2R\sin(2\psi)}{\sqrt{1-R^2}}.
```

Implementation:

```text
psrism/annual_anisotropy.py
```

Relevant functions:

```text
anisotropy_coefficients
effective_velocity_components
anisotropic_effective_speed
scintillation_velocity
```

These functions are currently library helpers rather than a full CLI MCMC fitter. A full annual anisotropy fit needs an epoch table and priors for `V_IISM,α`, `V_IISM,δ`, `A_r`, `ψ`, and `D_s`.

### Refractive Scintillation Estimates

`psrism` can derive scintillation quantities from a fitted scattering time. For each fitted subband, the decorrelation bandwidth implied by the scattering time is:

```math
\Delta\nu_{\rm d}
=
\frac{C_1}{2\pi\tau}.
```

The default value is:

```math
C_1 = 1.16.
```

If the pulsar distance `D` and effective transverse speed `V` are supplied, the diffractive scintillation timescale is estimated from:

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

The finite-scintle uncertainty is estimated using:

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

with defaults:

```math
\eta_t = 0.2,
\qquad
\eta_\nu = 0.2.
```

The finite-scintle contribution to the τ uncertainty is:

```math
\sigma_{\rm fse}
=
\frac{\tau}{\sqrt{N_{\rm scintles}}},
```

and the total uncertainty is:

```math
\sigma_{\rm total}
=
\sqrt{
\sigma_{\rm fit}^2
+
\sigma_{\rm fse}^2
}.
```

The refractive scintillation timescale is:

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
```

Run with isotropic subband τ fits:

```bash
psrism ARCHIVE --estimate-refractive --tau-subbands 4
```

This reports `Δν_d` from τ. Add distance and velocity to estimate `Δt_d`, `N_scintles`, `σ_fse`, `σ_total`, and `T_r`:

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
psrism/dynamic_flux.py                     Profile-window detection and dynamic-spectrum flux extraction
psrism/dynamic_spectrum.py                 Dynamic-spectrum construction and normalization
psrism/plot_dynamic_spectrum.py            Dynamic-spectrum and integrated-profile plotting
psrism/autocorrelation_spectrum.py         Autocorrelation-spectrum calculation
psrism/plot_autocorrelation_spectrum.py    Autocorrelation-spectrum plotting
psrism/scintillation_spectrum.py           Secondary-spectrum calculation
psrism/plot_scintillation_spectrum.py      Secondary-spectrum plotting
psrism/fit_secondary_spectrum.py           Parabolic-arc fitting
psrism/anisotropic_scattering.py           Anisotropic pulse-broadening fitting
psrism/annual_anisotropy.py                Annual anisotropy velocity-model helpers
psrism/refractive_scintillation.py         Derived scintillation and refractive estimates
psrism/fit_tau.py                          Scattered-pulse model and τ fitting
psrism/fit_alpha.py                        Frequency-scaling fit for α
psrism/plot_tau_vs_freq.py                 τ-versus-frequency plotting
psrism/time_series_analysis.py             Directory time-series CSV writing and plotting
psrism/bandpass.py                         Dynamic-spectrum bandpass scaling
psrism/dm_vs_time.py                       DM-versus-time helpers and plotting
psrism/tau_vs_time.py                      τ-versus-time helpers and plotting
psrism/fit_autocorrelation_spectrum.py     Autocorrelation-spectrum fitting
```

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
