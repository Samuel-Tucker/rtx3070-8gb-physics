#!/usr/bin/env python3
"""Measure Qwen3-0.6B weight-only quantisation quality on fixed raw text.

Requires PyTorch and the local HF snapshot. Runs full causal inference with
FP16 GEMMs on reconstructed weights. The CUDA benchmark measures packed kernels.
"""

import argparse
import functools
import json
import mmap
import os
import re
import struct
import tempfile
import time
import unicodedata
from pathlib import Path

import torch
from torch.nn import functional as F


SNAPSHOT = Path(
    "./Qwen3-0.6B"
    "snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
)
TRAIN_LAYERS = (0, 7, 14, 21, 27)
PROJECTIONS = {
    "q_proj": "self_attn",
    "k_proj": "self_attn",
    "v_proj": "self_attn",
    "o_proj": "self_attn",
    "gate_proj": "mlp",
    "up_proj": "mlp",
    "down_proj": "mlp",
}
BITS = {
    "fp16": 16.0,
    "int4": 4.25,
    "hsal_1lift": 1.125,
    "hsal_2lift": 2.25,
    "sol_dual_sign": 2.5,
    "sol_dual_sign_fwht": 2.5,
}
ROTATED = frozenset(("hsal_1lift", "hsal_2lift", "sol_dual_sign_fwht"))
BOOK_SIZE, CODE_DIM = 128, 8
ROTATION_SIZE, HSAL_GROUP, SCALAR_GROUP = 64, 128, 64
FIT_GROUPS, LLOYD_STEPS, SCORE_BATCH = 48, 15, 65536
SEED = 574103

PASSAGES = {
    # Charles Darwin, On the Origin of Species, introduction (1859).
    "prose": (
        "When on board H.M.S. Beagle, as naturalist, I was much struck with certain "
        "facts in the distribution of the inhabitants of South America, and in the "
        "geological relations of the present to the past inhabitants of that "
        "continent. These facts seemed to me to throw some light on the origin of "
        "species—that mystery of mysteries, as it has been called by one of our "
        "greatest philosophers. On my return home, it occurred to me, in 1837, "
        "that something might perhaps be made out on this question by patiently "
        "accumulating and reflecting on all sorts of facts which could possibly "
        "have any bearing on it."
    ),
    "python": '''from collections import deque


def shortest_distances(graph, source):
    """Return unweighted distances from source to every reachable node."""
    distances = {source: 0}
    pending = deque([source])
    while pending:
        node = pending.popleft()
        for neighbour in graph.get(node, ()):
            if neighbour in distances:
                continue
            distances[neighbour] = distances[node] + 1
            pending.append(neighbour)
    return distances
''',
}


class SafeWeights:
    def __init__(self, path):
        with path.open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            if length > 100_000_000 or length + 8 > path.stat().st_size:
                raise ValueError("Invalid safetensors header length")
            self.entries = json.loads(source.read(length))
            self.start = 8 + length
            self.data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_COPY)

    def load(self, name, device):
        entry = self.entries[name]
        dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[entry["dtype"]]
        start, end = entry["data_offsets"]
        count = (end - start) // dtype.itemsize
        if not (0 <= start < end <= len(self.data) - self.start):
            raise ValueError(f"Invalid safetensors offsets for {name}")
        tensor = torch.frombuffer(self.data, dtype=dtype, count=count, offset=self.start + start)
        return tensor.reshape(entry["shape"]).to(device=device, dtype=torch.float16, copy=True)


# Standard-library implementation of the exact Split + ByteLevel sequence
# stored in the pinned tokenizer. Build Unicode property classes for the
# characters in the string, preserving the order and backtracking of the regex.
SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
WHITESPACE = r"\x09-\x0d\x20\x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000"


@functools.lru_cache(maxsize=32)
def split_regex(non_ascii):
    letters = "A-Za-z" + "".join(re.escape(c) for c in non_ascii if unicodedata.category(c)[0] == "L")
    numbers = "0-9" + "".join(re.escape(c) for c in non_ascii if unicodedata.category(c)[0] == "N")
    return re.compile(
        rf"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n{letters}{numbers}]?[{letters}]+|[{numbers}]|"
        rf" ?[^{WHITESPACE}{letters}{numbers}]+[\r\n]*|[{WHITESPACE}]*[\r\n]+|"
        rf"[{WHITESPACE}]+(?![^{WHITESPACE}])|[{WHITESPACE}]+"
    )


class Tokenizer:
    def __init__(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        byte_level = {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False}
        split, byte = data["pre_tokenizer"]["pretokenizers"]
        model = data["model"]
        if (
            data["normalizer"] != {"type": "NFC"}
            or data["pre_tokenizer"]["type"] != "Sequence"
            or split != {"type": "Split", "pattern": {"Regex": SPLIT_PATTERN}, "behavior": "Isolated", "invert": False}
            or any(byte[key] != value for key, value in byte_level.items())
            or data["post_processor"]["type"] != "ByteLevel"
            or data["decoder"]["type"] != "ByteLevel"
            or model["type"] != "BPE"
            or any(model.get(key) for key in (
                "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "byte_fallback",
                "dropout", "ignore_merges",
            ))
        ):
            raise ValueError(f"Unsupported Qwen tokenizer pipeline in {path}")
        self.vocab = model["vocab"]
        self.tokens = {idx: token for token, idx in self.vocab.items()}
        self.ranks = {
            tuple(pair.split(" ") if isinstance(pair, str) else pair): rank
            for rank, pair in enumerate(model["merges"])
        }

        direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        remaining = sorted(set(range(256)) - set(direct))
        self.byte_encode = dict(zip(direct, map(chr, direct)))
        self.byte_encode.update({b: chr(256 + i) for i, b in enumerate(remaining)})
        self.byte_decode = {char: b for b, char in self.byte_encode.items()}

        self.added = {}
        for token in data["added_tokens"]:
            if any(token[key] for key in ("single_word", "lstrip", "rstrip", "normalized")):
                raise ValueError(f"Unsupported added-token flags for {token['content']}")
            self.added[token["content"]] = token["id"]
        self.added_ids = {idx: token for token, idx in self.added.items()}
        added = sorted(self.added, key=len, reverse=True)
        self.added_split = re.compile("(" + "|".join(map(re.escape, added)) + ")")

    @functools.lru_cache(maxsize=8192)
    def bpe(self, piece):
        symbols = [self.byte_encode[b] for b in piece.encode("utf-8")]
        while len(symbols) > 1:
            i = min(
                range(len(symbols) - 1),
                key=lambda i: self.ranks.get((symbols[i], symbols[i + 1]), len(self.ranks)),
            )
            pair = symbols[i], symbols[i + 1]
            if pair not in self.ranks:
                break
            symbols[i:i + 2] = ["".join(pair)]
        return tuple(self.vocab[symbol] for symbol in symbols)

    def encode(self, text):
        ids = []
        for span in self.added_split.split(text):
            if span in self.added:
                ids.append(self.added[span])
            elif span:
                normalized = unicodedata.normalize("NFC", span)
                non_ascii = "".join(sorted({c for c in normalized if ord(c) >= 128}))
                for match in split_regex(non_ascii).finditer(normalized):
                    ids.extend(self.bpe(match[0]))
        return ids

    def decode(self, ids):
        output = bytearray()
        for idx in ids:
            if idx in self.added_ids:
                output.extend(self.added_ids[idx].encode("utf-8"))
            else:
                token = self.tokens.get(idx)
                if token is None:
                    raise ValueError(f"Model generated ID {idx} outside the tokenizer vocabulary")
                output.extend(self.byte_decode[c] for c in token)
        return output.decode("utf-8", errors="replace")


def rms_norm(x, weight, eps):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype) * weight


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class Qwen:
    def __init__(self, config, weights, device):
        self.config = config
        self.device = device
        self.layers = config["num_hidden_layers"]
        self.n_heads = config["num_attention_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.eps = config["rms_norm_eps"]
        if (
            config["model_type"] != "qwen3"
            or self.layers != 28
            or config["hidden_act"] != "silu"
            or config["attention_bias"]
            or config.get("use_sliding_window")
            or config.get("rope_scaling") is not None
            or self.n_heads % self.n_kv_heads
        ):
            raise ValueError("Expected the full-attention, bias-free Qwen3-0.6B snapshot")
        self.embed = weights.load("model.embed_tokens.weight", device)
        self.head = self.embed if config["tie_word_embeddings"] else weights.load("lm_head.weight", device)
        self.final_norm = weights.load("model.norm.weight", device)
        self.body = []
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            names = {key: f"{module}.{key}.weight" for key, module in PROJECTIONS.items()}
            names.update({
                "input_norm": "input_layernorm.weight",
                "post_norm": "post_attention_layernorm.weight",
                "q_norm": "self_attn.q_norm.weight",
                "k_norm": "self_attn.k_norm.weight",
            })
            self.body.append({key: weights.load(prefix + name, device) for key, name in names.items()})

    def positional(self, length):
        dims = torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32)
        freq = self.config["rope_theta"] ** (-dims / self.head_dim)
        angle = torch.arange(length, device=self.device, dtype=torch.float32)[:, None] * freq
        angle = torch.cat((angle, angle), dim=-1)
        return angle.cos().half(), angle.sin().half()

    def attention(self, q, k, v, w, position):
        batch, length, _ = q.shape
        q = rms_norm(q.reshape(batch, length, self.n_heads, self.head_dim), w["q_norm"], self.eps)
        k = rms_norm(k.reshape(batch, length, self.n_kv_heads, self.head_dim), w["k_norm"], self.eps)
        v = v.reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = position
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        repeats = self.n_heads // self.n_kv_heads
        y = F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1),
            dropout_p=0.0, is_causal=True,
        )
        return y.transpose(1, 2).reshape(batch, length, -1)

    def forward_fp16(self, ids):
        x = F.embedding(ids, self.embed)
        position = self.positional(ids.size(1))
        for w in self.body:
            z = rms_norm(x, w["input_norm"], self.eps)
            q, k, v = (F.linear(z, w[name]) for name in ("q_proj", "k_proj", "v_proj"))
            x = x + F.linear(self.attention(q, k, v, w, position), w["o_proj"])
            z = rms_norm(x, w["post_norm"], self.eps)
            x = x + F.linear(F.silu(F.linear(z, w["gate_proj"])) * F.linear(z, w["up_proj"]), w["down_proj"])
        return x

    def logits(self, states):
        return F.linear(rms_norm(states, self.final_norm, self.eps), self.head).float()


def generation_check(model, tokenizer):
    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    generated = []
    eos = model.config["eos_token_id"]
    eos = {eos} if isinstance(eos, int) else set(eos)
    for _ in range(12):
        ids = torch.tensor([prompt_ids + generated], device=model.device)
        state = model.forward_fp16(ids)[:, -1:]
        token = int(model.logits(state).argmax(dim=-1).item())
        if token in eos:
            break
        generated.append(token)
    completion = tokenizer.decode(generated)
    # Open-ended coherence is subjective. Check a known answer and well-formed
    # text, and record the actual continuation and IDs for inspection.
    if (
        not re.match(r"\s*Paris\b", completion)
        or "\ufffd" in completion
        or any(c in completion for c in tokenizer.added)
        or any(unicodedata.category(c) == "Cc" and c not in "\n\r\t" for c in completion)
    ):
        raise RuntimeError(f"FP16 greedy generation smoke check failed: {prompt + completion!r}")
    return {"prompt": prompt, "continuation": completion, "greedy_token_ids": generated}


def hadamard(device):
    h = torch.ones((1, 1), dtype=torch.float16, device=device)
    for _ in range(ROTATION_SIZE.bit_length() - 1):
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h * ROTATION_SIZE ** -0.5


def fwht(x, transform, output_fp32=False):
    # For these fixed 64-point groups the dense orthonormal butterfly matrix
    # uses one tensor-core GEMM. Its binary coefficients are exact in FP16.
    groups = x.reshape(-1, ROTATION_SIZE)
    if output_fp32 and x.is_cuda:
        rotated = torch.mm(groups, transform, out_dtype=torch.float32)
    elif output_fp32:
        rotated = groups.float() @ transform.float()
    else:
        rotated = groups @ transform
    return rotated.reshape(x.shape)


def int4(weight):
    groups = weight.float().reshape(-1, SCALAR_GROUP)
    scale = (groups.abs().amax(dim=-1, keepdim=True) / 7.0).half().float()
    divisor = torch.where(scale == 0, 1.0, scale)
    levels = (groups / divisor).round().clamp_(-8, 7)
    return (levels * scale).reshape(weight.shape).half()


def dual_sign(weight):
    groups = weight.float().reshape(-1, SCALAR_GROUP)
    alpha1 = groups.abs().mean(dim=-1, keepdim=True).half().float()
    first = torch.where(groups < 0, -alpha1, alpha1)
    residual = groups - first
    alpha2 = residual.abs().mean(dim=-1, keepdim=True).half().float()
    second = torch.where(residual < 0, -alpha2, alpha2)
    return (first + second).reshape(weight.shape).half()


def normalized_groups(groups):
    scale = groups.square().mean(dim=-1, keepdim=True).sqrt().half().float()
    return groups / torch.where(scale == 0, 1.0, scale), scale


def lloyd(points, generator):
    """Fit 128 antipodal centroids by signed nearest-neighbour assignment."""
    initial = torch.randperm(points.size(0), generator=generator, device=points.device)[:BOOK_SIZE]
    centres = points[initial].contiguous()
    for _ in range(LLOYD_STEPS):
        dots = points @ centres.T
        objective = 2.0 * dots.abs() - centres.square().sum(-1)
        idx = objective.argmax(dim=-1)
        chosen = dots.gather(1, idx[:, None])
        aligned = torch.where(chosen < 0, -points, points)
        counts = torch.zeros(BOOK_SIZE, device=points.device).index_add_(0, idx, torch.ones_like(idx, dtype=points.dtype))
        sums = torch.zeros_like(centres).index_add_(0, idx, aligned)
        centres = torch.where(counts[:, None] > 0, sums / counts[:, None].clamp_min(1), centres)
    return centres.half()


class Codebook:
    def __init__(self, centres):
        self.values = torch.cat((centres, -centres)).contiguous()
        self.columns = self.values.T.contiguous()
        self.bias = (-0.5 * self.values.float().square().sum(dim=1)).half()

    def reconstruct(self, groups):
        normalized, scales = normalized_groups(groups)
        points = normalized.reshape(-1, CODE_DIM).half()
        restored = torch.empty_like(points)
        for start in range(0, points.size(0), SCORE_BATCH):
            stop = min(start + SCORE_BATCH, points.size(0))
            scores = torch.addmm(self.bias, points[start:stop], self.columns)
            chosen = scores.argmax(dim=-1)
            restored[start:stop] = self.values.index_select(0, chosen)
        return restored.reshape_as(groups).float() * scales


def fit_codebook(model, transform):
    cpu_rng = torch.Generator().manual_seed(SEED)
    fit_rng = torch.Generator(device=model.device).manual_seed(SEED)
    samples = []
    for layer in TRAIN_LAYERS:
        for name in PROJECTIONS:
            weight = model.body[layer][name].reshape(-1, HSAL_GROUP)
            index = torch.randperm(weight.size(0), generator=cpu_rng)[:FIT_GROUPS].to(model.device)
            samples.append(fwht(weight.index_select(0, index), transform, output_fp32=True))
    raw = torch.cat(samples, dim=0)
    first, _ = normalized_groups(raw)
    book = Codebook(lloyd(first.reshape(-1, CODE_DIM), fit_rng))
    return book, int(raw.numel() // CODE_DIM)


def quantized_weights(weight, transform, book):
    yield "fp16", weight
    yield "int4", int4(weight)
    yield "sol_dual_sign", dual_sign(weight)
    turned = fwht(weight, transform, output_fp32=True)
    yield "sol_dual_sign_fwht", dual_sign(turned)
    groups = turned.reshape(-1, HSAL_GROUP)
    first = book.reconstruct(groups)
    yield "hsal_1lift", first.half().reshape_as(weight)
    second = book.reconstruct(groups - first)
    yield "hsal_2lift", (first + second).half().reshape_as(weight)


def cosine(a, b, mask):
    return F.cosine_similarity((a.float() * mask).flatten(), (b.float() * mask).flatten(), dim=0)


def forward_all(model, ids, mask, transform, book):
    def prepare(inputs):
        return {
            method: fwht(activation, transform) if method in ROTATED else activation
            for method, activation in inputs.items()
        }

    x = dict.fromkeys(BITS, F.embedding(ids, model.embed))
    position = model.positional(ids.size(1))
    metrics = {method: [] for method in BITS}
    for index, w in enumerate(model.body):
        layer = {method: {"layer": index, "projection_cosine": {}} for method in BITS}

        def project(name, inputs):
            result = {}
            for method, weight in quantized_weights(w[name], transform, book):
                y = F.linear(inputs[method], weight)
                result[method] = y
                layer[method]["projection_cosine"][name] = cosine(result["fp16"], y, mask)
            return result

        z = {method: rms_norm(a, w["input_norm"], model.eps) for method, a in x.items()}
        z = prepare(z)
        q, k, v = (project(name, z) for name in ("q_proj", "k_proj", "v_proj"))
        attn = {method: model.attention(q[method], k[method], v[method], w, position) for method in BITS}
        o = project("o_proj", prepare(attn))
        x = {method: x[method] + o[method] for method in BITS}
        z = {method: rms_norm(a, w["post_norm"], model.eps) for method, a in x.items()}
        z = prepare(z)
        gate, up = project("gate_proj", z), project("up_proj", z)
        hidden = {method: F.silu(gate[method]) * up[method] for method in BITS}
        down = project("down_proj", prepare(hidden))
        x = {method: x[method] + down[method] for method in BITS}
        for method in BITS:
            layer[method]["layer_output_cosine"] = cosine(x["fp16"], x[method], mask)
            metrics[method].append(layer[method])
    return x, metrics


def to_metrics(layers):
    values = [
        value
        for layer in layers
        for value in (*layer["projection_cosine"].values(), layer["layer_output_cosine"])
    ]
    values = iter(torch.stack(values).cpu().tolist())
    return [
        {
            "layer": layer["layer"],
            "projection_cosine": {name: next(values) for name in layer["projection_cosine"]},
            "layer_output_cosine": next(values),
        }
        for layer in layers
    ]


def losses(model, states, ids, lengths):
    results = {}
    for batch, (name, length) in enumerate(lengths.items()):
        pieces = []
        for start in range(0, length - 1, 32):
            end = min(start + 32, length - 1)
            logits = model.logits(states[batch, start:end])
            pieces.append(F.cross_entropy(logits, ids[batch, start + 1:end + 1], reduction="sum"))
        results[name] = {
            "tokens_scored": length - 1,
            "total_nll_nats": float(torch.stack(pieces).sum().item()),
        }
        results[name]["nll_nats_per_token"] = results[name]["total_nll_nats"] / (length - 1)
    total = sum(row["total_nll_nats"] for row in results.values())
    count = sum(row["tokens_scored"] for row in results.values())
    return total / count, results


def write_json(path, result):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(result, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--output", type=Path, default=Path("track2_quality.json"))
    parser.add_argument("--device", default="cuda", help="PyTorch device, default cuda")
    args = parser.parse_args()
    start = time.perf_counter()
    device = torch.device(args.device)
    snapshot = args.snapshot.resolve()
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer(snapshot / "tokenizer.json")
    weights = SafeWeights(snapshot / "model.safetensors")
    try:
        model = Qwen(config, weights, device)
    finally:
        weights.data.close()

    generation = generation_check(model, tokenizer)
    print(f"FP16 greedy check  {generation['prompt'] + generation['continuation']!r}", flush=True)
    tokenized = {name: tokenizer.encode(text) for name, text in PASSAGES.items()}
    for name, ids in tokenized.items():
        if len(ids) < 2 or tokenizer.decode(ids) != unicodedata.normalize("NFC", PASSAGES[name]):
            raise ValueError(f"Tokenizer round-trip failed for {name}")
    lengths = {name: len(ids) for name, ids in tokenized.items()}
    width = max(lengths.values())
    if width > config["max_position_embeddings"]:
        raise ValueError("Evaluation exceeds the configured context length")
    pad = config["bos_token_id"]
    ids = torch.tensor([row + [pad] * (width - len(row)) for row in tokenized.values()], device=device)
    mask = torch.arange(width, device=device)[None, :] < torch.tensor(list(lengths.values()), device=device)[:, None]
    transform = hadamard(device)
    book, training_vectors = fit_codebook(model, transform)
    states, metrics = forward_all(model, ids, mask[..., None], transform, book)

    body_weights = sum(w[name].numel() for w in model.body for name in PROJECTIONS)
    results = {}
    for method, bpw in BITS.items():
        nll, by_passage = losses(model, states[method], ids, lengths)
        layers = to_metrics(metrics[method])
        projection_cos = [
            value for layer in layers for value in layer["projection_cosine"].values()
        ]
        codebook_bytes = BOOK_SIZE * CODE_DIM * 2 if method.startswith("hsal_") else 0
        results[method] = {
            "body_bits_per_weight": bpw,
            "body_payload_bytes": int(body_weights * bpw / 8),
            "shared_codebook_bytes": codebook_bytes,
            "nll_nats_per_token": nll,
            "delta_nll_vs_fp16": nll - (results["fp16"]["nll_nats_per_token"] if results else nll),
            "mean_projection_cosine": sum(projection_cos) / len(projection_cos),
            "min_projection_cosine": min(projection_cos),
            "mean_layer_output_cosine": sum(layer["layer_output_cosine"] for layer in layers) / len(layers),
            "passages": by_passage,
            "layers": layers,
        }

    summary = {
        "snapshot": str(snapshot),
        "config": config,
        "torch_version": torch.__version__,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - start,
        "greedy_check": generation,
        "scored_tokens": sum(length - 1 for length in lengths.values()),
        "evaluation_text": PASSAGES,
        "body_weights": body_weights,
        "projection_count": model.layers * len(PROJECTIONS),
        "fit": {
            "seed": SEED,
            "layers": list(TRAIN_LAYERS),
            "training_vectors": training_vectors,
            "lloyd_iterations": LLOYD_STEPS,
            "shared_fp16_centroids": BOOK_SIZE,
            "centroid_dimensions": CODE_DIM,
            "fwht_width": ROTATION_SIZE,
            "int4_zero_point": 8,
            "int4_and_sol_group_size": SCALAR_GROUP,
            "hsal_scale_group_size": HSAL_GROUP,
        },
        "cosine_basis": "Per-projection and block outputs versus FP16 on each method's own causal activations, excluding padding",
        "formats": results,
    }
    write_json(args.output, summary)
    print(f"\n{'Format':<21} {'bpw':>7} {'NLL':>9} {'Δ NLL':>9} {'Proj cos':>10} {'Block cos':>10}")
    for name, row in results.items():
        print(
            f"{name:<21} {row['body_bits_per_weight']:7.3f} {row['nll_nats_per_token']:9.4f} "
            f"{row['delta_nll_vs_fp16']:9.4f} {row['mean_projection_cosine']:10.6f} "
            f"{row['mean_layer_output_cosine']:10.6f}"
        )
    print(f"\n{summary['scored_tokens']} scored tokens, {summary['elapsed_seconds']:.2f} s, {args.output}")


if __name__ == "__main__":
    main()
