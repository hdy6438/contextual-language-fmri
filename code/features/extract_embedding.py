"""Extract causal decoder embeddings at the original story span boundaries."""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import tokenizers
import torch
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_story(characters_path, spans_path):
    characters, spans = read_tsv(characters_path), read_tsv(spans_path)
    if len(characters) != 869 or len(spans) != 732:
        raise ValueError("Expected 869 characters and 732 original spans.")
    for i, row in enumerate(characters, 1):
        if row["trial_type"] != "character" or len(row["text"]) != 1 or int(row["character_index"]) != i:
            raise ValueError(f"Invalid character event at row {i}.")
    onsets = [Decimal(row["onset"]) for row in characters]
    offsets = [t + Decimal(row["duration"]) for t, row in zip(onsets, characters, strict=True)]
    if (
        onsets[0] < 0
        or any(start >= end for start, end in zip(onsets, offsets, strict=True))
        or any(end > start for end, start in zip(offsets, onsets[1:]))
    ):
        raise ValueError("Expected ordered, non-overlapping character times.")
    story = "".join(row["text"] for row in characters)
    ends = []
    previous_end = 0
    for i, span in enumerate(spans):
        start, end = int(span["character_start"]) - 1, int(span["character_end"])
        if (
            int(span["embedding_row"]) != i
            or not previous_end == start < end <= len(story)
            or span["text"] != story[start:end]
            or Decimal(span["onset"]) != onsets[start]
            or Decimal(span["offset"]) != offsets[end - 1]
            or Decimal(span["duration"]) != offsets[end - 1] - onsets[start]
        ):
            raise ValueError(f"Original span text or timing changed at row {i}.")
        ends.append(end)
        previous_end = end
    if previous_end != len(story):
        raise ValueError("The original spans must cover the story.")
    return story, ends


def align_prefixes(story, ends, tokenizer):
    """Prove which standalone text prefixes share the full-story tokenization."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with character offsets is required.")
    if not ends or ends[-1] != len(story) or any(a >= b for a, b in zip([0, *ends[:-1]], ends, strict=True)):
        raise ValueError("Expected increasing span ends through the final character.")

    encoded = tokenizer(
        [story[:end] for end in ends], add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, padding=False, verbose=False,
    )
    sequences = encoded.input_ids
    reserved = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    for i, (ids, offsets, end) in enumerate(zip(sequences, encoded.offset_mapping, ends, strict=True)):
        if not ids or len(ids) != len(offsets) or reserved.intersection(ids):
            raise ValueError(f"Empty tokenization or unexpected reserved token at span row {i}.")
        covered = 0
        for start, stop in offsets:
            # Byte fallback can assign the same character to several tokens.
            # Some SentencePiece tokenizers add a zero-width marker at the start.
            if (start, stop) == (0, 0) and covered == 0:
                continue
            if not 0 <= start < stop <= end or start > covered:
                raise ValueError(f"Invalid or discontinuous token offsets at span row {i}.")
            covered = max(covered, stop)
        if covered != end:
            raise ValueError(f"Token offsets do not cover the text through span row {i}.")

    full_ids, full_offsets = sequences[-1], encoded.offset_mapping[-1]
    # A byte overlap or merge spanning this boundary also requires an
    # independent prefix pass, even if the prefix happens to have matching IDs.
    reuse = []
    for end, ids in zip(ends, sequences, strict=True):
        n = len(ids)
        reuse.append(
            full_ids[:n] == ids
            and all(stop <= end for _, stop in full_offsets[:n])
            and all(start >= end for start, _ in full_offsets[n:])
        )
    return sequences, reuse, full_offsets


class _CapturedHidden(Exception):
    def __init__(self, hidden):
        self.hidden = hidden


def layer_stack(model, n_layers):
    for container in (model, getattr(model, "decoder", None)):
        for name in ("layers", "h"):
            blocks = getattr(container, name, None)
            if isinstance(blocks, torch.nn.ModuleList) and len(blocks) == n_layers:
                return blocks
    raise ValueError("Could not find the decoder blocks for --stop-at-layer.")


def middle_hidden(model, inputs, mask, layer, block=None):
    if block is None:
        result = model(input_ids=inputs, attention_mask=mask, use_cache=False, output_hidden_states=True)
        return result.hidden_states[layer]

    def capture(_block, _arguments, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("Unexpected decoder block output.")
        raise _CapturedHidden(hidden)

    hook = block.register_forward_hook(capture)
    try:
        model(input_ids=inputs, attention_mask=mask, use_cache=False, output_hidden_states=False)
    except _CapturedHidden as captured:
        return captured.hidden
    finally:
        hook.remove()
    raise RuntimeError("The selected decoder block did not run.")


def forward_prefixes(model, sequences, pad, layer, block=None):
    lengths = [len(ids) for ids in sequences]
    tokens = torch.full((len(sequences), max(lengths)), pad, dtype=torch.long)
    attention = torch.zeros_like(tokens)
    for i, ids in enumerate(sequences):
        tokens[i, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        attention[i, :len(ids)] = 1
    return middle_hidden(model, tokens, attention, layer, block)


def representative_rows(rows):
    if not rows:
        return []
    return sorted({rows[0], rows[len(rows) // 2], rows[-1]})


@torch.inference_mode()
def extract_embeddings(model, sequences, reuse, pad, layer, batch_size, block=None):
    shared = [i for i, can_reuse in enumerate(reuse) if can_reuse]
    separate = sorted((i for i, can_reuse in enumerate(reuse) if not can_reuse), key=lambda i: len(sequences[i]))
    if block is not None:
        sample = [sequences[-1][:min(32, len(sequences[-1]))]]
        torch.testing.assert_close(
            forward_prefixes(model, sample, pad, layer, block),
            forward_prefixes(model, sample, pad, layer),
            atol=4e-4, rtol=2e-4,
        )

    hidden = forward_prefixes(model, [sequences[-1]], pad, layer, block)
    if len(shared) == 1 and len(sequences[-1]) > 1:
        # Even without a shareable earlier span, check an earlier token.
        token_stop = len(sequences[-1]) // 2
        standalone = forward_prefixes(model, [sequences[-1][:token_stop]], pad, layer, block)
        torch.testing.assert_close(hidden[0, token_stop - 1], standalone[0, -1], atol=4e-4, rtol=2e-4)
    result = torch.empty((len(sequences), hidden.shape[-1]), dtype=torch.float32)
    result[shared] = hidden[0, [len(sequences[i]) - 1 for i in shared]].float()
    del hidden
    for start in range(0, len(separate), batch_size):
        batch = separate[start:start + batch_size]
        hidden = forward_prefixes(model, [sequences[i] for i in batch], pad, layer, block)
        result[batch] = hidden[range(len(batch)), [len(sequences[i]) - 1 for i in batch]].float()
        del hidden
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(separate):
            print(f"Finished {min(start + batch_size, len(separate))}/{len(separate)} prefix-specific passes.", flush=True)

    # Compare the shared full-story extraction and the padded fallback batch
    # against standalone, exactly retokenized and unpadded prefixes.
    checked = representative_rows(shared[:-1]) + representative_rows(separate)
    for i in checked:
        standalone = forward_prefixes(model, [sequences[i]], pad, layer, block)[0, -1].float()
        torch.testing.assert_close(result[i], standalone, atol=4e-4, rtol=2e-4)
    return result.numpy(), checked


def context_limit(config, tokenizer):
    for name in ("max_position_embeddings", "n_positions", "max_sequence_length", "max_seq_len", "seq_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and 0 < value < 10**9:
            return value
    value = tokenizer.model_max_length
    return value if isinstance(value, int) and 0 < value < 10**9 else None


def load_text_decoder(checkpoint, attention=None):
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    options = {"attn_implementation": attention} if attention else {}
    if config.model_type == "gemma3":
        # Transformers has used both legacy and current Gemma 3 VLM key prefixes.
        # Loading only the nested text config skips the vision tower and projector.
        options["key_mapping"] = {r"^(?:model\.)?language_model(?:\.model)?\.": ""}
        config = config.get_text_config()
    elif config.model_type == "gemma3_text":
        # The standalone CausalLM writes model.* but the bare text decoder does not.
        options["key_mapping"] = {r"^(?:model|language_model(?:\.model)?)\.": ""}
    if config.model_type == "gemma3_text" and config.use_bidirectional_attention:
        raise ValueError("Expected a Gemma 3 decoder configured for causal text attention.")

    model, info = AutoModel.from_pretrained(
        checkpoint, config=config, local_files_only=True, dtype=torch.float32,
        output_loading_info=True, **options,
    )
    for field in ("missing_keys", "mismatched_keys", "error_msgs"):
        if info[field]:
            raise ValueError(f"Invalid decoder checkpoint ({field}): {info[field][:5]}")
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(os.getenv("PAPER_INPUTS", "data/inputs")))
    parser.add_argument("--output", type=Path, default=Path(os.getenv("PAPER_OUTPUT", "outputs")) / "features")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.model_id = "google/gemma-3-1b-pt"
    args.revision = "fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
    args.output_stem = "embeddings"
    args.attention, args.stop_at_layer = "sdpa", True
    if args.threads < 1 or args.batch_size < 1:
        parser.error("--threads and --batch-size must be positive.")
    if not args.model_dir.is_dir():
        parser.error("--model-dir must be an existing local checkpoint directory.")
    if Path(args.output_stem).name != args.output_stem or args.output_stem in {"", ".", ".."}:
        parser.error("--output-stem must be a filename stem.")

    characters = args.inputs / "character_events.tsv"
    spans = args.inputs / "spans.tsv"
    story, ends = read_story(characters, spans)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    sequences, reuse, offsets = align_prefixes(story, ends, tokenizer)
    full_tokens = sequences[-1]
    unknown = [
        {"character_start": start + 1, "character_end": stop, "text": story[start:stop]}
        for token, (start, stop) in zip(full_tokens, offsets, strict=True)
        if token == tokenizer.unk_token_id
    ]
    print(
        f"{len(ends)} original spans, {len(full_tokens)} full-story tokens, "
        f"{len(ends) - sum(reuse)} prefix-specific passes, {len(unknown)} full-story unknown tokens.",
        flush=True,
    )

    torch.set_num_threads(args.threads)
    model = load_text_decoder(args.model_dir, args.attention)
    config = model.config
    depth = config.num_hidden_layers
    if depth != 26 or config.hidden_size != 1152:
        raise ValueError("Expected the fixed Gemma 3 1B text checkpoint (26 blocks, 1152 dimensions).")
    if config.is_encoder_decoder or depth < 2:
        raise ValueError("Expected a causal model with at least two decoder blocks.")
    layer = depth // 2
    block = layer_stack(model, depth)[layer - 1] if args.stop_at_layer else None
    bos = config.bos_token_id if config.bos_token_id is not None else tokenizer.bos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (bos if bos is not None else 0)
    if bos is not None:
        sequences = [[bos, *ids] for ids in sequences]
    limit = context_limit(config, tokenizer)
    if limit is not None and max(map(len, sequences)) > limit:
        raise ValueError(f"A complete story prefix exceeds the model context of {limit} tokens.")
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if not 0 <= pad < vocab_size or any(not 0 <= token < vocab_size for ids in sequences for token in ids):
        raise ValueError("The tokenizer and model embedding vocabulary do not match.")
    print(f"Extracting decoder block {layer}/{depth} on CPU.", flush=True)
    embedding, checked = extract_embeddings(model, sequences, reuse, pad, layer, args.batch_size, block)
    if embedding.shape[0] != len(ends) or embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("Invalid extracted embedding matrix.")

    args.output.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output / f"{args.output_stem}.npy"
    np.save(embedding_path, embedding, allow_pickle=False)
    metadata = {
        "model": args.model_id,
        "revision": args.revision,
        "decoder_block": layer,
        "decoder_blocks": depth,
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "alignment": "Final token of the exact text prefix ending at each original span, in embedding_row order.",
        "context": "Continuous causal story prefix. Identical token prefixes share one pass.",
        "start_token_id": bos,
        "story_tokens": len(full_tokens),
        "model_input_tokens": len(sequences[-1]),
        "max_positions": limit,
        "retokenized_prefix_count": len(ends) - sum(reuse),
        "retokenized_prefix_rows": [i for i, can_reuse in enumerate(reuse) if not can_reuse],
        "verified_prefix_rows": checked,
        "tokenizer": type(tokenizer).__name__,
        "unknown_token_count": len(unknown),
        "unknown_tokens": unknown,
        "attention": config._attn_implementation,
        "stop_at_decoder_block": args.stop_at_layer,
        "characters_sha256": hashlib.sha256(characters.read_bytes()).hexdigest(),
        "spans_sha256": hashlib.sha256(spans.read_bytes()).hexdigest(),
        "embedding_sha256": hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
        "software": {"torch": torch.__version__, "transformers": transformers.__version__, "tokenizers": tokenizers.__version__},
    }
    (args.output / f"{args.output_stem}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved {embedding.shape} to {embedding_path}", flush=True)


if __name__ == "__main__":
    main()
