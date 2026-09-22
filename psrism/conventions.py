"""Scientific conventions shared across PSRISM analyses."""

from __future__ import annotations

import math


# Reference: Filothodoros thesis (ALEX.pdf), PDF pp. 34-35, Section 2.7.4,
# reports the power-law fit and interpolation of scattering time at 150 MHz.
TAU_REFERENCE_FREQUENCY_MHZ = 150.0

# Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 4.2.3,
# gives tau_s proportional to f**(-alpha) and the positive-alpha convention.
SCATTERING_INDEX_CONVENTION = "tau(nu) = tau_ref * (nu / nu_ref)^(-alpha)"

# Reference: Lorimer & Kramer (2005), psrhandbook.pdf, Section 7.4.4.1,
# defines the decorrelation bandwidth at half maximum and timescale at 1/e.
ACF_DECORRELATION_LEVEL = 0.5
ACF_TIMESCALE_LEVEL = math.exp(-1.0)
