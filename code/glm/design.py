"""Original HRF, speech controls and 316-frame joint text-GLM design."""
import math
from pathlib import Path
import subprocess
import numpy as np
from paths import read_tsv
TR = 0.75
N_VOLUMES = 336
N_PREFIX_SPANS = 32
HIGH_PASS_SECONDS = 128.0
FEATURES = ("surprisal", "entropy", "semantic_update")
SPEECH = ("event_density", "span_duration", "span_tokens", "audio_rms")
MOTION = [f"{axis}{suffix}" for axis in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
          for suffix in ("", "_derivative1", "_power2", "_derivative1_power2")]
TISSUE = ("white_matter", "csf")
def values(rows, key):
    return np.array([float(row[key]) if row[key] != "n/a" else np.nan for row in rows])

def double_gamma(t):
    lag = np.clip(t, 0, 32)
    response = np.exp(-lag) * (lag**5 / math.gamma(6) - lag**15 / (6 * math.gamma(16)))
    response[t > 32] = 0
    return response

HRF_PEAK = double_gamma(np.linspace(0, 32, 32001)).max()

def hrf_at(times, events):
    return double_gamma(times[:, None] - events[None, :]) / HRF_PEAK

def audio_envelope(path, ffmpeg):
    sample_rate = 16_000
    frame_samples = 320
    pcm = subprocess.run(
        [str(ffmpeg), "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(sample_rate), "-c:a", "pcm_f32le", "-f", "f32le", "-"],
        check=True, capture_output=True,
    )
    samples = np.frombuffer(pcm.stdout, dtype="<f4")
    if not np.isfinite(samples).all() or samples.size / sample_rate < 251.160:
        raise ValueError("Invalid or incomplete listening audio.")
    n_frames = samples.size // frame_samples
    frames = samples[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    step = frame_samples / sample_rate
    return (np.arange(n_frames) + 0.5) * step, np.sqrt(np.mean(frames**2, axis=1)) * step

def individual_design(common, confounds, start):
    all_rows = read_tsv(confounds)
    if len(all_rows) != N_VOLUMES:
        raise ValueError(f"Unexpected number of confound rows in {confounds}")
    rows = all_rows[start:]
    n = len(rows)
    fd = values(rows, "framewise_displacement")
    dvars = values(rows, "std_dvars")
    non_steady = [k for k in rows[0] if k.startswith("non_steady_state_outlier")]
    flags = (fd > 0.5) | (dvars > 1.5)
    for col in non_steady:
        flags |= values(rows, col) == 1
    if not np.isfinite(fd).all() or not np.isfinite(dvars).all():
        raise ValueError(f"Missing FD or standardized DVARS in retained rows of {confounds}")
    spike_frames = np.flatnonzero(flags)
    n_drifts = math.floor(2 * n * TR / HIGH_PASS_SECONDS)
    drifts = np.cos(np.pi * (np.arange(n)[:, None] + 0.5) * np.arange(1, n_drifts + 1) / n)

    names = [
        *FEATURES, *SPEECH, *MOTION, *TISSUE, *[f"dct_{j:02}" for j in range(1, n_drifts + 1)],
        *[f"spike_frame_{start + frame + 1:03}" for frame in spike_frames], "intercept",
    ]
    design = np.column_stack([
        common, *[values(rows, col) for col in MOTION + list(TISSUE)],
        drifts, np.eye(n)[:, spike_frames], np.ones(n),
    ])
    if len(names) != 37 + len(spike_frames) or design.shape != (316, len(names)):
        raise ValueError("Design matrix does not have the expected dimensions.")
    if not np.isfinite(design).all():
        raise ValueError(f"Non-finite design value for {confounds}")
    scaled = design[:, :-1] - design[:, :-1].mean(axis=0)
    norms = np.linalg.norm(scaled, axis=0)
    if np.any(norms == 0):
        raise ValueError(f"Constant non-intercept column for {confounds}")
    scaled /= norms
    rank = int(np.linalg.matrix_rank(scaled)) + 1
    if rank != design.shape[1]:
        raise ValueError(f"Rank-deficient design for {confounds}")
    vif = np.linalg.inv(scaled.T @ scaled).diagonal()[:len(FEATURES)]
    return names, design, len(spike_frames), rank, n - rank, vif


def speech_design(spans_path, audio_path, ffmpeg="ffmpeg"):
    """Return all 336 rows of the original four shared speech controls.

    Span token counts are from the canonical segmentation table, not the
    Gemma-native token completion counts used to score scalar features.
    """
    events = read_tsv(spans_path)
    if len(events) != 732 or not np.array_equal(values(events, "embedding_row"), np.arange(732)):
        raise ValueError("Expected 732 ordered canonical spans.")
    onset, offset, duration = (values(events, k) for k in ("onset", "offset", "duration"))
    if not np.allclose(offset - onset, duration) or np.any(onset[1:] < offset[:-1]):
        raise ValueError("Invalid canonical span timing.")
    times = np.arange(N_VOLUMES) * TR
    onset_hrf, offset_hrf = hrf_at(times, onset), hrf_at(times, offset)
    tokens = values(events, "token_end") - values(events, "token_start") + 1
    if not np.isfinite(tokens).all() or np.any(tokens < 1):
        raise ValueError("Canonical token counts must be positive and finite.")
    audio_times, rms = audio_envelope(audio_path, ffmpeg)
    return np.column_stack([
        (onset_hrf.sum(axis=1) + offset_hrf.sum(axis=1)) / 2,
        offset_hrf @ (duration - duration.mean()),
        offset_hrf @ (tokens - tokens.mean()),
        hrf_at(times, audio_times) @ rms,
    ])


def text_design(events_path, spans_path):
    """Three feature columns at 15.0--251.25 s; prior 32 spans excluded."""
    spans, reference = read_tsv(events_path), read_tsv(spans_path)
    keys = ("embedding_row", "onset", "duration", "offset", "text")
    if len(spans) != 732 or [tuple(r[k] for k in keys) for r in spans] != [tuple(r[k] for k in keys) for r in reference]:
        raise ValueError("Scalar features and canonical spans must share exactly the same timing.")
    times = np.arange(N_VOLUMES) * TR
    onset = values(spans, "onset")
    if int(np.searchsorted(times, onset[N_PREFIX_SPANS])) != 20 or not math.isclose(onset[N_PREFIX_SPANS], 14.360):
        raise ValueError("Expected first retained volume at 15.0 s after span 33 starts at 14.360 s.")
    onset_hrf, offset_hrf = (hrf_at(times, values(spans, key)) for key in ("onset", "offset"))
    columns = []
    for key, basis in zip(("surprisal_bits", "entropy_bits", "semantic_update"), (offset_hrf, onset_hrf, offset_hrf)):
        col = values(spans, key)
        critical = col[N_PREFIX_SPANS:]
        if not np.isfinite(critical).all() or critical.std() <= 0:
            raise ValueError(f"Invalid {key}.")
        if key == "semantic_update" and not np.isnan(col[:N_PREFIX_SPANS]).all():
            raise ValueError("Semantic update requires 32 preceding spans.")
        columns.append(basis[:, N_PREFIX_SPANS:] @ ((critical - critical.mean()) / critical.std()))
    return np.column_stack(columns)[20:]
