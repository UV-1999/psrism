import csv
import json
from pathlib import Path

from psrism.batch_analysis import (
    BatchPulsarRun,
    discover_archive_groups,
    write_batch_exports,
)


class _Archive:
    def __init__(self, source):
        self.source = source

    def get_source(self):
        return self.source


def test_discovery_groups_archives_by_header_source_and_records_failures(tmp_path):
    paths = [tmp_path / "one.nop", tmp_path / "two.nop", tmp_path / "bad.nop"]
    sources = {"one.nop": "J1234+5678", "two.nop": "J1234+5678"}

    def loader(path):
        name = Path(path).name
        if name == "bad.nop":
            raise RuntimeError("unreadable archive")
        return _Archive(sources[name])

    result = discover_archive_groups(paths, loader)

    assert tuple(result.groups) == ("J1234+5678",)
    assert result.groups["J1234+5678"] == tuple(paths[:2])
    assert result.failures[0].archive_path.endswith("bad.nop")


def test_discovery_can_filter_selected_sources(tmp_path):
    paths = [tmp_path / "a.nop", tmp_path / "b.nop"]

    def loader(path):
        return _Archive("J0001+0001" if Path(path).name == "a.nop" else "J0002+0002")

    result = discover_archive_groups(paths, loader, {"j0002+0002"})

    assert tuple(result.groups) == ("J0002+0002",)


def test_discovery_normalizes_psr_prefix_and_ignores_calibrators(tmp_path):
    pulsar = tmp_path / "pulsar.nop"
    calibrator = tmp_path / "calibrator.nop"

    def loader(path):
        source = "PSR J1234+5678" if Path(path).name == pulsar.name else "3C147OFF"
        return _Archive(source)

    result = discover_archive_groups([pulsar, calibrator], loader)

    assert tuple(result.groups) == ("J1234+5678",)
    assert result.ignored_non_pulsars[0].source == "3C147OFF"


def test_batch_exports_combine_epochs_and_make_pulsar_summary(tmp_path):
    archives = (tmp_path / "a.nop", tmp_path / "b.nop")
    series_path = tmp_path / "J1234+5678_time_series.csv"
    with series_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "archive_path",
                "archive_name",
                "mjd",
                "observing_frequency_mhz",
                "period_s",
                "dm",
                "alpha",
                "alpha_error",
                "tau0_s",
                "tau0_error_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "archive_path": str(archives[0]),
                "archive_name": archives[0].name,
                "mjd": "60000",
                "observing_frequency_mhz": "150",
                "period_s": "1",
                "dm": "10",
                "alpha": "4",
                "alpha_error": "0.2",
                "tau0_s": "0.1",
                "tau0_error_s": "0.01",
            }
        )
        writer.writerow(
            {
                "archive_path": str(archives[1]),
                "archive_name": archives[1].name,
                "mjd": "60010",
                "observing_frequency_mhz": "160",
                "period_s": "1",
                "dm": "12",
                "alpha": "3.8",
                "alpha_error": "0.1",
                "tau0_s": "0.2",
                "tau0_error_s": "0.02",
            }
        )
    discovery = discover_archive_groups(
        list(archives),
        lambda _path: _Archive("J1234+5678"),
    )
    run = BatchPulsarRun(
        pulsar="J1234+5678",
        archive_paths=archives,
        exit_code=0,
        output_directory=tmp_path,
        time_series_csv=series_path,
        terminal_log=tmp_path / "run.log",
    )

    exports = write_batch_exports(
        [run],
        discovery,
        tmp_path / "sample_batch",
        input_root=tmp_path,
        pattern="*.nop",
        recursive=False,
        selected_sources=[],
        requested_parameters="dm,alpha",
    )

    with exports.observations_csv.open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))
    with exports.pulsars_csv.open(newline="", encoding="utf-8") as handle:
        summary = next(csv.DictReader(handle))
    manifest = json.loads(exports.manifest_json.read_text(encoding="utf-8"))

    assert len(observations) == 2
    assert observations[0]["pulsar"] == "J1234+5678"
    assert summary["epochs_processed"] == "2"
    assert summary["dm_median"] == "11.0"
    assert summary["time_span_days"] == "10.0"
    assert summary["alpha_weighted_mean"] == "3.84"
    assert summary["tau_ref_fraction_period"] == "0.12"
    assert manifest["epochs_exported"] == 2
    assert exports.scattering_comparison_plot is not None
