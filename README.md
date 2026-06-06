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

## Plotting

All heatmap-style plots use the `afmhot` color map. The plot title is the archive file name.

### Dynamic Spectrum

The dynamic spectrum shows pulse intensity as a function of observing time and radio frequency. It is the primary input for scintillation analysis.

For each subintegration and frequency channel, `psrism` estimates an on-pulse flux from the noise-normalized pulse profile:

$$
E(t,\nu)=
\max\left[
0,\,
\frac{1}{N_{\rm on}}
\int_{\phi_{\rm on}}
\frac{P(t,\nu,\phi)-\mu_{\rm off}(t,\nu)}
{\sigma_{\rm off}(t,\nu)}
d\phi
\right].
$$

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
plot_autocorrelation_spectrum
```

Run:

```bash
psrism ARCHIVE --acspec
```

The output file is:

```text
*_acspec.png
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

The FFT-shifted axes are made with `numpy.fft.fftfreq`. The x-axis is fringe frequency in Hz, and the y-axis is delay in seconds. The zero fringe-frequency row is set to zero before display.

The calculation is implemented in [`psrism/scintillation_spectrum.py`](psrism/scintillation_spectrum.py), and plotting with attached marginals is implemented in [`psrism/plot_scintillation_spectrum.py`](psrism/plot_scintillation_spectrum.py).

Relevant functions:

```text
calculate_scintillation_spectrum
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

The τ-versus-frequency plot also includes a comparison line with fixed `α = -4.4`, normalized to the fitted `τ0` at the same reference frequency.

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
psrism/fit_tau.py                          Scattered-pulse model and τ fitting
psrism/fit_alpha.py                        Frequency-scaling fit for α
psrism/plot_tau_vs_freq.py                 τ-versus-frequency plotting
psrism/bandpass.py                         Dynamic-spectrum bandpass scaling
psrism/dm_vs_time.py                       DM-versus-time helpers and plotting
psrism/tau_vs_time.py                      τ-versus-time helpers and plotting
psrism/fit_autocorrelation_spectrum.py     Autocorrelation-spectrum fitting
```

## Quick Fixes

### Matplotlib cache warning

If matplotlib reports that `~/.config/matplotlib` is not writable, set a writable cache directory before running plots:

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
