# this is test line for push pull working.
# psrism

CLI-oriented Python package for pulsar ISM analysis.

The original monolithic script is copied as `psrism.py` for reference. The package code lives in `psrism/`.

## Install

From this directory:

```bash
python -m pip install -e .
```

`psrchive` is also required for archive loading and is usually installed separately from the Python package manager.

If matplotlib warns that `~/.config/matplotlib` is not writable, set a writable cache directory before running plots:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-$USER
```

## CLI

```bash
psrism ARCHIVE --dspec --acspec --sspec --intpf
```

When `--intpf` is used, `psrism` also fits the integrated profile with the scatter-broadened model, prints tau with its uncertainty, and overlays the model on `*_intpf.png`.

When `--acspec` is used, `psrism` reports the ACF decorrelation bandwidth measured at `ACF(delta_f, 0) = 1/2` and the diffractive timescale measured at `ACF(0, tau) = 1/e`.

Useful processing options:

```bash
psrism ARCHIVE --nsub 64 --nchan 128 --nbin 512 --dm 9.0 --onw 10
```

For quick testing in this workspace, avoid the corrupted `J0814+7429_2025-10-16_02:31:00.lane0b.00.nop` file. PSRCHIVE cannot load it. Use a readable archive such as:

```bash
psrism /home/piyushmarmat/old/J0139+5814_2019-08-07_03:31:00.lane0b.00.nop --nsub 50 --nchan 61 --nbin 256 --dspec
```

Scrunch targets must divide the current archive dimensions. For the `J0139` file, `vap` reports:

```text
nsub=1050, nchan=366, nbin=1024
```

So `--nsub 50`, `--nchan 61`, and `--nbin 256` are valid, while `--nchan 32` is not.

You can inspect an archive before running `psrism` with:

```bash
vap -c file,nsub,npol,nchan,nbin,dm,freq,bw ARCHIVE
```

Or use the built-in inspection mode:

```bash
psrism ARCHIVE --inspect
```

This prints the archive dimensions and valid smaller scrunch targets for `--nsub`, `--nchan`, and `--nbin`. PSRCHIVE scrunch targets must divide the current dimension exactly.

To measure tau in frequency subbands and fit the scattering spectral index alpha:

```bash
psrism ARCHIVE --fit-alpha --tau-subbands 4
```

This fits an exponentially modified Gaussian profile in each contiguous frequency subband, then fits:

```text
log10(tau) = alpha * log10(nu / nu0) + log10(tau0)
```

using weighted least squares with `sigma_log_tau = sigma_tau / (tau * ln(10))`. It prints the subband tau values and saves `*_tau_vs_freq.png`.

It also saves `*_subband_tau_fits.png`, containing one profile-plus-fit overlay panel for each frequency subband.

## Dynamic Spectrum

For each subintegration and frequency channel, `psrism` estimates an on-pulse flux from the noise-normalized profile:

$$
E(t,\nu) =
\max\left[
0,\,
\frac{1}{N_{\rm on}}
\int_{\phi_{\rm on}}
\frac{P(t,\nu,\phi)-\mu_{\rm off}(t,\nu)}
{\sigma_{\rm off}(t,\nu)}
d\phi
\right].
$$

Here $\mu_{\rm off}$ and $\sigma_{\rm off}$ are estimated from off-pulse phase bins. The dynamic spectrum plot shows $E(t,\nu)$ with attached time and frequency marginals.

## Autocorrelation Spectrum

`--acspec` computes the finite-lag covariance function following the Cordes/Lorimer-Kramer convention:

$$
{\rm CF}(\Delta\nu,\tau) =
\sum_{\nu=1}^{n_\nu-|\Delta\nu|}
\sum_{t=1}^{n_t-|\tau|}
E(\nu,t)\,E(\nu+\Delta\nu,t+\tau),
$$

and

$$
{\rm ACF}(\Delta\nu,\tau)=
\frac{{\rm CF}(\Delta\nu,\tau)}
{{\rm CF}(0,0)}.
$$

The implementation subtracts the mean dynamic-spectrum level before calculating the covariance and uses zero-padded linear correlation, not circular FFT autocorrelation. The returned/plot axes use integer lags in the range `-N/2 < lag < N/2`.

The decorrelation bandwidth and diffractive timescale are reported as:

$$
\Delta\nu_{\rm DISS}: {\rm ACF}(\Delta\nu,0)=\frac{1}{2},
$$

and

$$
\Delta t_{\rm DISS}: {\rm ACF}(0,\tau)=\frac{1}{e}.
$$

## Secondary Spectrum

The secondary spectrum is calculated from the mean-subtracted dynamic spectrum:

$$
E'(t,\nu)=E(t,\nu)-\langle E\rangle,
$$

$$
S(f_t,f_\nu)=
\left|
{\cal F}_2\{E'(t,\nu)\}
\right|^2.
$$

For plotting, the code uses:

$$
S_{\rm dB}=10\log_{10}\left(S+10^{-12}\right),
$$

with FFT-shifted axes from `numpy.fft.fftfreq`. The x-axis is fringe frequency in Hz and the y-axis is delay in seconds. The zero fringe-frequency row is set to zero before display.

## Tau And Alpha Fitting

The integrated-profile and subband tau fits use an exponentially modified Gaussian scattered-pulse model:

$$
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
$$

For `--fit-alpha`, the archive bandwidth is divided into contiguous frequency subbands. The representative subband frequency is the midpoint of that subband:

$$
\nu_i =
\frac{\nu_{\rm min,i}+\nu_{\rm max,i}}{2}.
$$

Each subband profile is fit independently to obtain $(\tau_i,\sigma_{\tau_i})$. The scattering spectral index is then fit in log space:

$$
x_i=\log_{10}\left(\frac{\nu_i}{\nu_0}\right),
\qquad
y_i=\log_{10}(\tau_i),
$$

$$
\sigma_{y_i} =
\frac{\sigma_{\tau_i}}
{\tau_i\ln 10},
$$

and weighted least squares is used for:

$$
y_i = \alpha x_i + b.
$$

Finally,

$$
\tau_0 = 10^b.
$$

The tau-versus-frequency plot also shows a comparison line with fixed $\alpha=-4.4$, normalized to the fitted $\tau_0$ at the same reference frequency.

## Module Map

- `psrism/cli.py`: command line interface.
- `psrism/archive_io.py`: archive loading, preprocessing, and reduction to NumPy arrays.
- `psrism/dynamic_flux.py`: profile-window detection and dynamic-spectrum flux extraction.
- `psrism/dynamic_spectrum.py`: dynamic spectrum construction and normalization.
- `psrism/plot_dynamic_spectrum.py`: dynamic spectrum and integrated-profile plotting.
- `psrism/autocorrelation_spectrum.py`: autocorrelation spectrum calculation.
- `psrism/plot_autocorrelation_spectrum.py`: autocorrelation spectrum plotting.
- `psrism/scintillation_spectrum.py`: scintillation/secondary spectrum calculation.
- `psrism/plot_scintillation_spectrum.py`: scintillation/secondary spectrum plotting.
- `psrism/fit_tau.py`: scattered-pulse model and tau fitting.
- `psrism/fit_alpha.py`: tau-frequency power-law alpha fitting.
- `psrism/plot_tau_vs_freq.py`: tau versus frequency plotting.
- `psrism/bandpass.py`: dynamic-spectrum bandpass scaling.
- `psrism/dm_vs_time.py`: DM versus time helpers and plotting.
- `psrism/tau_vs_time.py`: tau versus time helpers and plotting.
- `psrism/fit_autocorrelation_spectrum.py`: autocorrelation spectrum fitting.
