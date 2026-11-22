"""Extract native Gemma 3 1B surprisal, onset entropy and 32-span semantic update."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from paths import ROOT, MODEL, INPUTS, OUTPUT, CACHE, EVENTS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from extract_embedding import align_prefixes, context_limit, read_story, read_tsv
WINDOW = 32
REPO = "google/gemma-3-1b-pt"
REVISION = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def entropy(logits):
    lp = logits.double().log_softmax(dim=-1)
    return -(lp.exp() * lp).sum().item() / math.log(2)

@torch.inference_mode()
def token_features(model, sequences, reuse, bos):
    full = sequences[-1]
    inp = torch.tensor([[bos, *full]], dtype=torch.long)
    # Each of the first N logit rows predicts one observed story token.
    result = model(inp, use_cache=False, logits_to_keep=torch.arange(len(full)))
    vocab_size = model.config.get_text_config().vocab_size
    if result.logits.shape != (1, len(full), vocab_size):
        raise ValueError(f"Expected N vocabulary distributions, got {result.logits.shape}.")
    surprise, pretoken_entropy = score_tokens(result.logits[0], full)
    del result
    at_onset = np.empty(len(sequences))
    at_onset[0] = pretoken_entropy[0]
    for i, (prefix, shared) in enumerate(zip(sequences[:-1], reuse[:-1], strict=True), 1):
        if shared:
            at_onset[i] = pretoken_entropy[len(prefix)]
        else:
            indices = torch.tensor([[bos, *prefix]], dtype=torch.long)
            at_onset[i] = entropy(model(indices, use_cache=False, logits_to_keep=1).logits[0, 0])
    # Verify a common boundary against the standalone text prefix.
    common = [i for i in range(len(sequences) - 1) if reuse[i]]
    if common:
        i = common[len(common) // 2]
        indices = torch.tensor([[bos, *sequences[i]]], dtype=torch.long)
        ref = entropy(model(indices, use_cache=False, logits_to_keep=1).logits[0, 0])
        np.testing.assert_allclose(at_onset[i + 1], ref, rtol=2e-4, atol=2e-4)
    if (not np.isfinite(surprise).all() or np.any(surprise < 0)
            or not np.isfinite(at_onset).all() or np.any(at_onset < 0)
            or np.any(at_onset > math.log2(vocab_size) + 1e-10)):
        raise ValueError("Invalid surprisal or entropy.")
    return surprise, at_onset

def aggregate_surprisal(values, offsets, span_ends, story_length):
    ends = np.array([stop for _, stop in offsets], dtype=int)
    if not np.all(np.diff(ends) >= 0) or ends[-1] != story_length:
        raise ValueError("Expected nondecreasing full-story token offsets ending at the last character.")
    # Offset-zero tokenizer markers have no character/time and are excluded.
    tokens = np.flatnonzero(ends > 0)
    rows = np.searchsorted(span_ends, ends[tokens], side="left")
    sums = np.bincount(rows, weights=values[tokens], minlength=len(span_ends))
    counts = np.bincount(rows, minlength=len(span_ends))
    bounds = np.cumsum(np.r_[0, counts]) + tokens[0]
    np.testing.assert_allclose(sums.sum(), values[tokens].sum(), rtol=5e-15, atol=1e-12)
    return sums, counts, bounds, int(len(ends) - len(tokens))

def write_text_atomic(path, text):
    temp = path.with_suffix(".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)
def score_tokens(logits, ids):
    surprisal, entropy = [], []
    for start in range(0, len(ids), 64):
        stop = min(start + 64, len(ids))
        # With BOS prepended, logit row j predicts original story token j.
        log_p = logits[start:stop].double().log_softmax(dim=-1)
        actual = torch.tensor(ids[start:stop], dtype=torch.long).unsqueeze(1)
        surprisal.extend((-log_p.gather(1, actual)[:, 0] / math.log(2)).tolist())
        entropy.extend((-(log_p.exp() * log_p).sum(dim=1) / math.log(2)).tolist())
    return np.asarray(surprisal), np.asarray(entropy)

def semantic_update(embeddings):
    unit = embeddings.astype(np.float64)
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    updates = np.full(len(unit), np.nan)
    for i in range(WINDOW, len(unit)):
        preceding = unit[i - WINDOW:i]
        before = preceding.mean(axis=0)
        after = (preceding.sum(axis=0) + unit[i]) / (len(preceding) + 1)
        cosine = np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))
        updates[i] = 1 - np.clip(cosine, -1, 1)
    return updates


def load_model(checkpoint):
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if getattr(config.get_text_config(), "use_bidirectional_attention", False):
        raise ValueError("Expected causal attention.")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, local_files_only=True, dtype=torch.float32,
        attn_implementation="sdpa", output_loading_info=True,
    )
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        if info[field]:
            raise ValueError(f"Invalid causal checkpoint ({field}): {info[field][:5]}")
    return model.eval(), tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CACHE / "checkpoints" / MODEL,
                        help="Locally downloaded, fixed Gemma 3 1B checkpoint.")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    characters_path, spans_path = INPUTS / "character_events.tsv", INPUTS / "spans.tsv"
    story, ends = read_story(characters_path, spans_path)
    spans = read_tsv(spans_path)
    embedding_path = OUTPUT / "features/embeddings.npy"
    meta = json.loads(embedding_path.with_suffix(".json").read_text())
    for key, expected in (("model", REPO), ("revision", REVISION),
                          ("characters_sha256", digest(characters_path)), ("spans_sha256", digest(spans_path)),
                          ("embedding_sha256", digest(embedding_path))):
        if meta[key] != expected:
            raise ValueError(f"Embedding metadata mismatch: {key}")
    embedding = np.load(embedding_path, allow_pickle=False)
    if embedding.shape != (732, 1152) or not np.isfinite(embedding).all() or np.any(np.linalg.norm(embedding, axis=1) == 0) or meta["decoder_block"] != 13:
        raise ValueError("Expected 732 Gemma block-13 embeddings of width 1152.")
    update = semantic_update(embedding)
    if not np.isnan(update[:WINDOW]).all() or not np.isfinite(update[WINDOW:]).all():
        raise ValueError("Unexpected semantic update support.")
    model, tokenizer = load_model(args.checkpoint)
    config = model.config.get_text_config()
    sequences, reuse, offsets = align_prefixes(story, ends, tokenizer)
    full = sequences[-1]
    bos = config.bos_token_id if config.bos_token_id is not None else tokenizer.bos_token_id
    max_context = context_limit(config, tokenizer)
    unknown = [i for i, token in enumerate(full) if token == tokenizer.unk_token_id]
    if (bos is None or bos != meta["start_token_id"] or len(full) != meta["story_tokens"]
            or len(unknown) != meta["unknown_token_count"]
            or len(sequences) - sum(reuse) != meta.get("retokenized_prefix_count", 0)
            or config.vocab_size != model.get_output_embeddings().weight.shape[0]
            or (max_context is not None and max(map(len, sequences)) + 1 > max_context)):
        raise ValueError("Tokenizer/checkpoint disagrees with the embedding source.")
    surprisal, at_onset = token_features(model, sequences, reuse, bos)
    total, counts, bounds, skipped = aggregate_surprisal(surprisal, offsets, ends, len(story))
    destination = ROOT / "features" / MODEL
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / EVENTS
    metadata = {
        "model": REPO, "revision": REVISION, "device": "CPU", "dtype": "float32",
        "start_token_id": bos, "vocab_size": config.vocab_size, "span_count": len(spans),
        "canonical_spans_sha256": digest(spans_path), "characters_sha256": digest(characters_path),
        "embedding_sha256": meta["embedding_sha256"], "decoder_block": meta["decoder_block"],
        "native_story_token_count": len(full), "n_zero_width_tokens_excluded_from_surprisal": skipped,
        "n_spans_without_native_token_completion": int((counts == 0).sum()),
        "native_unknown_token_count": len(unknown), "native_unknown_tokens": meta["unknown_tokens"],
        "surprisal_bits": "Sum native token surprisal once at the offset of the span containing its final character.",
        "entropy_bits": "Native next-token Shannon entropy at exact preceding text prefix; retokenize crossed boundaries.",
        "semantic_update": "Cosine update of preceding 32 L2-normalized contextual vectors; undefined in first 32 spans.",
    }
    write_text_atomic(path.with_suffix(".json"), json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    columns = list(spans[0])
    for column in ("surprisal_bits", "entropy_bits", "semantic_update"):
        if column not in columns:
            columns.append(column)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for i, original in enumerate(spans):
            span = dict(original)
            span["token_start"] = str(bounds[i] + 1) if counts[i] else "n/a"
            span["token_end"] = str(bounds[i + 1]) if counts[i] else "n/a"
            span.update(surprisal_bits=f"{total[i]:.12g}", entropy_bits=f"{at_onset[i]:.12g}",
                        semantic_update="n/a" if i < WINDOW else f"{update[i]:.12g}")
            writer.writerow(span)
    tmp.replace(path)
    print(path)


if __name__ == "__main__":
    main()
