"""Cheap, deterministic Stage-1 features for causal policy context."""

from __future__ import annotations

import numpy as np

BANDS_HZ = ((0, 300), (300, 1200), (1200, 3000), (3000, None))


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-8)))


def extract_stage1_features(
    waveform: np.ndarray,
    sample_rate: int,
    noise_floor_db: float,
    previous_spectrum: np.ndarray | None = None,
) -> tuple[dict[str, float], np.ndarray]:
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    rms_db = _db(float(np.sqrt(np.mean(wave * wave))) if wave.size else 0.0)
    spectrum = np.abs(np.fft.rfft(wave * np.hanning(len(wave)))) if wave.size else np.zeros(1)
    frequencies = np.fft.rfftfreq(len(wave), 1 / sample_rate) if wave.size else np.zeros(1)
    total = max(float(np.square(spectrum).sum()), 1e-12)
    features: dict[str, float] = {
        "rms_db": rms_db,
        "noise_floor_db": float(noise_floor_db),
        "delta_db": rms_db - float(noise_floor_db),
        "spectral_flux": 0.0
        if previous_spectrum is None or previous_spectrum.shape != spectrum.shape
        else float(np.square(np.maximum(spectrum - previous_spectrum, 0)).sum() / total),
    }
    for low, high in BANDS_HZ:
        mask = frequencies >= low
        if high is not None:
            mask &= frequencies < high
        features[f"band_energy_{low}_{high or 'nyquist'}"] = float(
            np.square(spectrum[mask]).sum() / total
        )
    features["s1_a_trigger"] = float(features["delta_db"] >= 6.0)
    features["s1_b_trigger"] = float(
        features["s1_a_trigger"] or features["spectral_flux"] >= 0.08
    )
    features["s1_c_trigger"] = float(
        features["s1_b_trigger"]
        or max(features[name] for name in features if name.startswith("band_energy_")) >= 0.55
    )
    return features, spectrum
