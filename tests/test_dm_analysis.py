import json
import subprocess

import numpy as np
import pytest

from psrism.dm_analysis import (
    DISPERSION_CONSTANT_MHZ2_S,
    run_pdmp,
    scattering_dm_from_subbands,
    write_dm_report,
)


def test_run_pdmp_parses_result_and_builds_requested_command(tmp_path, monkeypatch):
    archive = tmp_path / "epoch.ar"
    archive.write_bytes(b"archive")
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Best S/N = 42.5\n"
                "Best BC Period (ms) = 1000.2 Error (ms) = 0.3\n"
                "Best DM = 49.125 Correction = 0.025 Error = 0.004\n"
            ),
        )

    monkeypatch.setattr("psrism.dm_analysis.shutil.which", lambda _name: "/usr/bin/pdmp")
    monkeypatch.setattr("psrism.dm_analysis.subprocess.run", fake_run)

    result = run_pdmp(
        archive,
        archive_dm=49.1,
        dm_half_range=0.5,
        dm_step=0.01,
        max_channels=64,
        max_subints=32,
        max_bins=256,
    )

    assert result.best_dm == pytest.approx(49.125)
    assert result.dm_correction == pytest.approx(0.025)
    assert result.dm_error == pytest.approx(0.004)
    assert result.best_snr == pytest.approx(42.5)
    assert result.best_barycentric_period_ms == pytest.approx(1000.2)
    assert result.period_error_ms == pytest.approx(0.3)
    assert captured["command"] == [
        "/usr/bin/pdmp",
        "-f",
        "-g",
        "/null",
        "-dr",
        "0.5",
        "-ds",
        "0.01",
        "-mc",
        "64",
        "-ms",
        "32",
        "-mb",
        "256",
        str(archive.resolve()),
    ]
    assert captured["kwargs"]["check"] is False


def test_run_pdmp_does_not_accept_zero_exit_without_best_dm(tmp_path, monkeypatch):
    archive = tmp_path / "epoch.ar"
    archive.write_bytes(b"archive")
    monkeypatch.setattr("psrism.dm_analysis.shutil.which", lambda _name: "/usr/bin/pdmp")
    monkeypatch.setattr(
        "psrism.dm_analysis.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="Error: TEMPO environment variable is not set\n",
        ),
    )

    with pytest.raises(RuntimeError, match="did not produce a Best DM result"):
        run_pdmp(archive, archive_dm=49.1)


def test_scattering_dm_recovers_synthetic_residual():
    frequency_mhz = np.array([120.0, 135.0, 150.0, 170.0, 190.0])
    period_s = 1.0
    nbin = 2048
    input_dm = 49.1
    expected_delta_dm = 0.023
    reference_mhz = 150.0
    arrival_s = 0.2 + DISPERSION_CONSTANT_MHZ2_S * expected_delta_dm * (
        frequency_mhz**-2 - reference_mhz**-2
    )
    phase_bins = arrival_s * nbin / period_s

    result = scattering_dm_from_subbands(
        frequency_mhz,
        phase_bins,
        period_s,
        nbin,
        input_dm,
        phase_error_bins=np.full(len(frequency_mhz), 0.05),
        reference_frequency_mhz=reference_mhz,
    )

    assert result.delta_dm == pytest.approx(expected_delta_dm, rel=1e-10)
    assert result.corrected_dm == pytest.approx(input_dm + expected_delta_dm)
    assert result.n_points == len(frequency_mhz)
    assert result.rms_residual_s < 1e-12


def test_dm_report_omits_raw_pdmp_console_output(tmp_path, monkeypatch):
    archive = tmp_path / "epoch.ar"
    archive.write_bytes(b"archive")
    monkeypatch.setattr("psrism.dm_analysis.shutil.which", lambda _name: "/usr/bin/pdmp")
    monkeypatch.setattr(
        "psrism.dm_analysis.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout="Best DM = 2.1 Correction = 0.1 Error = 0.01\n",
        ),
    )
    pdmp_result = run_pdmp(archive, archive_dm=2.0)
    output = tmp_path / "dm.json"

    write_dm_report(output, pdmp_result, None)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["pdmp"]["best_dm"] == pytest.approx(2.1)
    assert "stdout" not in payload["pdmp"]
    assert payload["scattering_dm"] is None
