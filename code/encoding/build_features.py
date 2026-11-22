import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


TR = .75
FIRST_FRAME = 12
FULL_FRAMES = 336
CONTEXT_SECONDS = (0., 1.5, 4.5)
FIR_LAG_SECONDS = (2., 4., 6., 8.)
LANCZOS_LOBES = 3


def build_features(embeddings, offsets, times, column_name):
    scores = embeddings - embeddings.mean(axis=0)
    cumulative = np.vstack([np.zeros((1, scores.shape[1])), scores.cumsum(axis=0)])
    stop_index = np.arange(1, len(scores) + 1)
    contexts = []
    for width in CONTEXT_SECONDS:
        if width == 0:
            contexts.append(scores)
            continue
        start_index = np.searchsorted(offsets, offsets - width, side="right")
        average = (cumulative[stop_index] - cumulative[start_index]) / (stop_index - start_index)[:, None]
        contexts.append(average)
    columns, labels, windows = [], [], []
    for lag in FIR_LAG_SECONDS:
        distance = (times[:, None] - lag - offsets) / TR
        # Center the same Lanczos interpolant at the exact FIR shift. TR is 0.75 s.
        weights = np.sinc(distance) * np.sinc(distance / LANCZOS_LOBES)
        weights[(np.abs(distance) >= LANCZOS_LOBES) | (offsets > times[:, None])] = 0
        for width, event_features in zip(CONTEXT_SECONDS, contexts, strict=True):
            columns.append(weights @ event_features)
            labels.extend(f"fir{lag:g}s_ctx{width:g}s_{column_name}{k:04}"
                          for k in range(1, embeddings.shape[1] + 1))
        windows.append({"fir_lag_s": lag, "n_rows_without_event_support": int((~weights.any(axis=1)).sum())})
    raw = np.column_stack(columns)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("A stimulus feature column is constant.")
    standardized = (raw - mean) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("Nonfinite encoding features.")
    return standardized, labels, windows, mean, scale


def save_design(output, scores, offsets, times, metadata, column_name):
    features, names, windows, mean, scale = build_features(scores, offsets, times, column_name)
    output.mkdir(parents=True, exist_ok=True)
    left, singular, right = np.linalg.svd(features, full_matrices=False)
    # Centering leaves one null direction in time. Keep every other direction,
    # preserving the full standardized feature Gram matrix and ridge penalty.
    if not singular[-2] > singular[0] * max(features.shape) * np.finfo(float).eps:
        raise ValueError("Expected a full-rank design within the centered time subspace.")
    coordinates = left[:, :-1] * singular[:-1]
    np.testing.assert_allclose(coordinates @ right[:-1], features, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(coordinates @ coordinates.T, features @ features.T, atol=2e-8, rtol=1e-10)
    np.save(output / "features.npy", coordinates, allow_pickle=False)
    np.save(output / "expanded_features.npy", features, allow_pickle=False)
    np.savez_compressed(output / "feature_scaling.npz", feature_mean=mean, feature_sd=scale)
    metadata.update({
        "n_events": len(offsets),
        "feature_names": names, "fir_lags_seconds": list(FIR_LAG_SECONDS), "resampling_support": windows,
        "context_width_seconds": list(CONTEXT_SECONDS),
        "n_expanded_features": len(names), "n_features": coordinates.shape[1],
        "ridge_coordinates": "Keep every nonzero singular direction in the time axis of the centered design. This exactly preserves the kernel of all standardized embedding features.",
        "context": "Use the current span at width zero, otherwise average embeddings available in the preceding window ending at each span offset, including the current span.",
        "lanczos": "Three-lobe Nyquist low-pass window, sinc(u) * sinc(u/3) with u = (t - FIR lag - event offset)/TR. Drop contributions from events after BOLD time t.",
        "lanczos_cutoff_hz": 1 / (2 * TR),
        "edge_handling": "Use observed story spans only; frames with no contributing spans have zero raw feature value.",
        "feature_standardization": "Center and scale each lag-context feature column over all retained stimulus time points",
        "tr_seconds": TR, "first_time_seconds": times[0], "last_time_seconds": times[-1], "n_frames": len(times),
    })
    (output / "features.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{features.shape[0]} time points, {features.shape[1]} expanded columns, {coordinates.shape[1]} exact ridge coordinates from {scores.shape[1]} {column_name} dimensions.")


def main():
    parser = argparse.ArgumentParser()
    root = Path(os.getenv("PAPER_OUTPUT", "outputs"))
    parser.add_argument("--events", type=Path, default=Path(os.getenv("PAPER_INPUTS", "data/inputs")) / "spans.tsv")
    parser.add_argument("--embeddings", type=Path, default=root / "features/embeddings.npy")
    parser.add_argument("--metadata", type=Path, default=root / "features/embeddings.json")
    parser.add_argument("--output", type=Path, default=root / "encoding")
    args = parser.parse_args()
    with args.events.open(encoding="utf-8") as stream:
        events = list(csv.DictReader(stream, delimiter="\t"))
    source = json.loads(args.metadata.read_text(encoding="utf-8"))
    embeddings = np.load(args.embeddings, allow_pickle=False).astype(np.float64)
    if embeddings.ndim != 2 or len(embeddings) != 732 or embeddings.shape[1] < 128 or not np.isfinite(embeddings).all():
        raise ValueError("Expected 732 finite contextual embeddings of dimension at least 128.")
    if [int(event["embedding_row"]) for event in events] != list(range(len(embeddings))):
        raise ValueError("Stimulus and embeddings must agree in row order.")
    offsets = np.array([float(event["offset"]) for event in events])
    if not np.isfinite(offsets).all() or not np.all(np.diff(offsets) > 0):
        raise ValueError("Stimulus offsets must be ordered, unique and finite.")
    centered = embeddings - embeddings.mean(axis=0)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    ratios = singular**2 / np.sum(singular**2)
    loadings = right[:128].copy()
    loadings *= np.sign(loadings[np.arange(128), np.argmax(np.abs(loadings), axis=1)])[:, None]
    times = np.arange(FIRST_FRAME, FULL_FRAMES) * TR
    original = {
        "embedding_source": source,
        "embedding_dimensions": embeddings.shape[1],
        "embedding_sha256": hashlib.sha256(args.embeddings.read_bytes()).hexdigest(),
        "event_sha256": hashlib.sha256(args.events.read_bytes()).hexdigest(),
    }
    save_design(args.output / "pc128", centered @ loadings.T, offsets, times, {
        **original, "embedding_pca": True, "n_components": 128,
        "pca_explained_variance": float(ratios[:128].sum()),
        "pc_count_for_95_percent_original_variance": int(np.searchsorted(np.cumsum(ratios), .95) + 1),
    }, "pc")
    np.savez_compressed(args.output / "pc128" / "embedding_pca.npz", loadings=loadings,
                        embedding_mean=embeddings.mean(axis=0), variance_ratio=ratios)
    print(f"Top 128 variance {ratios[:128].sum():.6%}, "
          f"{np.searchsorted(np.cumsum(ratios), .95) + 1} PCs for 95%.", flush=True)


if __name__ == "__main__":
    main()
