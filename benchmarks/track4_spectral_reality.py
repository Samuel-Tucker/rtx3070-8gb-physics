#!/usr/bin/env python3
"""Layer-14 Qwen3-0.6B projection substitutions. Requires CUDA PyTorch and the checkpoint.

Run: ~/tm-venv/bin/python track4_spectral_reality.py
The report measures single-token CUDA Graph latency and held-out real activations.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F


CHECKPOINT = Path(
    "./Qwen3-0.6B"
    "c1899de289a04d12100db370d81485cdf75e47ca/model.safetensors"
)
LAYER = 14
SEED = 3070
GROUP = 128
BLOCKS = 32
FIT_STEPS = 64
FIT_BATCH = 64
HEAD_DIM = 128
Q_HEADS = 16
KV_HEADS = 8
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000
PROJECTIONS = {
    "q_proj": "self_attn.q_proj",
    "v_proj": "self_attn.v_proj",
    "down_proj": "mlp.down_proj",
    "gate_proj": "mlp.gate_proj",
}
DTYPES = {"F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2), "F32": (torch.float32, 4)}

# Original short English passages, tokenised with the specified snapshot's Qwen
# tokenizer. Each row starts a fresh causal sequence. IDs are embedded so only
# model.safetensors is needed at runtime.
CALIBRATION_IDS = (
    # weather: The overnight forecast called for rain along the coast.
    (
        785, 24398, 17595, 2598, 369, 11174, 3156, 279, 13648, 13, 3216, 6556, 279, 9956,
        1030, 6519, 10984, 11, 9380, 2797, 49293, 916, 279, 6290, 323, 3347, 9437, 3941,
        279, 34131, 13, 576, 74642, 7341, 12433, 279, 2297, 19620, 1573, 279, 1156, 31631,
        2115, 13,
    ),
    # software: A database transaction groups several operations so that they succeed or fail together.
    (
        32, 4625, 7745, 5203, 3807, 7525, 773, 429, 807, 11996, 476, 3690, 3786, 13,
        1416, 279, 3633, 21025, 1283, 458, 2647, 11, 279, 2943, 1265, 1779, 279, 7745,
        1102, 1573, 22683, 287, 13, 1527, 877, 3262, 354, 2251, 1376, 15354, 279, 3538,
        43817, 264, 11504, 1681, 13,
    ),
    # fiction: Mara put the brass key on the windowsill and listened for footsteps on the stairs.
    (
        44, 5059, 2182, 279, 36038, 1376, 389, 279, 11030, 483, 323, 33693, 369, 59702,
        389, 279, 31149, 13, 576, 3753, 1030, 1012, 4287, 2474, 7589, 13, 3197, 279,
        9780, 8866, 10497, 11, 1340, 12771, 705, 1059, 22875, 323, 14858, 700, 1526, 279,
        13551, 13,
    ),
    # engineering: A pressure gauge measures the difference between the vessel and the surrounding air.
    (
        32, 7262, 30890, 10953, 279, 6672, 1948, 279, 26182, 323, 279, 14590, 3720, 13,
        13032, 279, 30298, 31430, 1573, 17592, 279, 30890, 13, 4636, 26345, 279, 13723, 11,
        1787, 279, 31430, 13970, 323, 1779, 279, 10284, 369, 36696, 13,
    ),
    # transport: Passengers for the early train should use the bridge at the west end of the platform.
    (
        12187, 14886, 369, 279, 4124, 5426, 1265, 990, 279, 14164, 518, 279, 9710, 835,
        315, 279, 5339, 13, 576, 11893, 29388, 279, 11727, 5163, 374, 1660, 51734, 13,
        16951, 646, 30893, 458, 10555, 6021, 369, 58968, 879, 1184, 3019, 1910, 2615, 13,
    ),
    # ecology: The survey followed a stream from the edge of the village to its source.
    (
        785, 10572, 8110, 264, 4269, 504, 279, 6821, 315, 279, 14126, 311, 1181, 2530,
        13, 576, 2083, 12864, 33139, 448, 5538, 27501, 323, 29994, 40811, 518, 1817, 2747,
        13, 9959, 9315, 16009, 45373, 1380, 279, 12408, 1030, 1012, 22949, 13,
    ),
    # arithmetic: A shop orders 24 boxes with 18 items in each box.
    (
        32, 8061, 10163, 220, 17, 19, 14697, 448, 220, 16, 23, 3589, 304, 1817,
        3745, 13, 29948, 3589, 525, 20123, 323, 2441, 220, 18, 20, 525, 738, 15663,
        369, 264, 3037, 13, 576, 1372, 2500, 369, 6278, 374, 279, 4024, 2790, 27283,
        2176, 5203, 13,
    ),
    # instructions: Place the bowl on a flat surface and weigh the flour before adding water.
    (
        17371, 279, 19212, 389, 264, 10063, 7329, 323, 17529, 279, 19828, 1573, 7842, 3015,
        13, 19219, 3080, 902, 9058, 28660, 7146, 11, 1221, 3421, 279, 19212, 323, 5274,
        279, 30352, 311, 2732, 13, 5985, 279, 80936, 8669, 1393, 279, 23387, 76562, 13,
    ),
)

HELD_OUT_IDS = (
    # administration: The council published the minutes on Friday afternoon.
    (
        785, 14833, 4652, 279, 4420, 389, 6602, 13354, 13, 9043, 40593, 1030, 1012, 11666,
        2337, 279, 6438, 11, 2670, 264, 2297, 311, 279, 8568, 4115, 13, 576, 31728,
        66540, 686, 387, 3208, 311, 279, 6733, 1573, 279, 1790, 2906, 13257, 13,
    ),
    # physics: Light bends when it passes between materials with different refractive indices.
    (
        13911, 92797, 979, 432, 16211, 1948, 7236, 448, 2155, 19353, 3028, 14937, 13, 362,
        18299, 16262, 8991, 518, 458, 9210, 34192, 803, 13970, 323, 4344, 5106, 13, 576,
        3311, 315, 57118, 13798, 389, 279, 9210, 323, 389, 279, 5888, 315, 279, 1378,
        7236, 13,
    ),
    # fiction: Elias unfolded the letter on the train and ran his thumb across the faded seal.
    (
        36, 71829, 75682, 279, 6524, 389, 279, 5426, 323, 10613, 806, 24050, 3941, 279,
        53334, 25349, 13, 576, 91709, 572, 11285, 11, 3498, 566, 1410, 537, 6099, 279,
        1537, 882, 566, 1030, 3884, 432, 13, 1260, 1349, 279, 1156, 1555, 10917, 13,
    ),
    # workshop: Measure the shaft at several points with a micrometer.
    (
        32236, 279, 30368, 518, 3807, 3501, 448, 264, 19197, 87271, 13, 1416, 279, 23033,
        34227, 3156, 1181, 3084, 11, 3255, 279, 7772, 323, 24632, 39074, 13, 9590, 279,
        28990, 12300, 1156, 1576, 264, 2613, 16392, 646, 2297, 279, 1102, 13,
    ),
    # accounting: An invoice records what was supplied, the date of supply and the amount due.
    (
        2082, 24615, 7424, 1128, 572, 17221, 11, 279, 2400, 315, 8149, 323, 279, 3311,
        4152, 13, 362, 8651, 6668, 5185, 646, 4396, 458, 1465, 1283, 279, 24615, 702,
        1012, 10897, 13, 13655, 2176, 7424, 773, 279, 27075, 646, 387, 50300, 2937, 13,
    ),
    # medicine: The hospital moved the appointment from Monday to Thursday.
    (
        785, 8777, 7726, 279, 17635, 504, 7014, 311, 7794, 13, 576, 23117, 380, 10774,
        279, 501, 882, 323, 4588, 279, 8720, 311, 4446, 279, 44780, 6524, 13, 576,
        27813, 19809, 374, 389, 279, 4910, 6422, 29388, 279, 1887, 23117, 18010, 13,
    ),
)


class SafeTensorReader:
    def __init__(self, path):
        self.file = path.open("rb")
        try:
            size = os.fstat(self.file.fileno()).st_size
            prefix = self.file.read(8)
            if len(prefix) != 8:
                raise ValueError("missing safetensors header length")
            header_size, = struct.unpack("<Q", prefix)
            if not 2 <= header_size <= min(100_000_000, size - 8):
                raise ValueError(f"invalid safetensors header size: {header_size}")
            self.header = json.loads(self.file.read(header_size))
            if not isinstance(self.header, dict):
                raise ValueError("invalid safetensors header")
            self.data_start = 8 + header_size
            self.data_size = size - self.data_start
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()

    def metadata(self, name):
        entry = self.header[name]
        dtype, width = DTYPES[entry["dtype"]]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        if (
            not shape or any(type(dim) is not int or dim <= 0 for dim in shape)
            or not 0 <= start < end <= self.data_size
            or end - start != width * math.prod(shape)
        ):
            raise ValueError(f"invalid tensor size or offsets: {name}")
        return dtype, width, shape, start, end

    def read(self, dtype, start, size):
        self.file.seek(self.data_start + start)
        data = bytearray(size)
        if self.file.readinto(data) != size:
            raise ValueError(f"truncated tensor data at offset {start}")
        return torch.frombuffer(data, dtype=dtype)

    def tensor(self, name, dtype):
        stored, _, shape, start, end = self.metadata(name)
        return self.read(stored, start, end - start).reshape(shape).to("cuda", dtype=dtype)

    def embeddings(self, name, ids):
        dtype, width, (rows, cols), start, _ = self.metadata(name)
        unique, inverse = ids.unique(sorted=True, return_inverse=True)
        keys = unique.tolist()
        if keys[0] < 0 or keys[-1] >= rows:
            raise ValueError("token ID outside the embedding table")
        stride = cols * width
        values = torch.stack([self.read(dtype, start + i * stride, stride) for i in keys])
        return values.to("cuda", dtype=torch.bfloat16)[inverse.to("cuda")]


def rms_norm(x, weight):
    xf = x.float()
    normed = xf * (xf.square().mean(dim=-1, keepdim=True) + RMS_EPS).rsqrt()
    return normed.to(x.dtype) * weight


def rotary(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-right, left), dim=-1) * sin


@torch.no_grad()
def capture_activations(reader):
    documents = CALIBRATION_IDS + HELD_OUT_IDS
    length = max(map(len, documents))
    ids = torch.tensor([doc + (0,) * (length - len(doc)) for doc in documents])
    mask = torch.arange(length)[None, :] < torch.tensor(list(map(len, documents)))[:, None]
    mask = mask.to("cuda")
    hidden = reader.embeddings("model.embed_tokens.weight", ids)
    inv_freq = ROPE_THETA ** (-torch.arange(0, HEAD_DIM, 2, device="cuda").float() / HEAD_DIM)
    angles = torch.outer(torch.arange(length, device="cuda").float(), inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(hidden.dtype), angles.sin().to(hidden.dtype)
    batch = len(documents)

    def split(x):
        split_at = len(CALIBRATION_IDS)
        return x[:split_at][mask[:split_at]].float(), x[split_at:][mask[split_at:]].float()

    for layer in range(LAYER + 1):
        prefix = f"model.layers.{layer}."

        def weight(name):
            return reader.tensor(prefix + name + ".weight", hidden.dtype)

        def project(x, name):
            return F.linear(x, weight(name))

        x = rms_norm(hidden, weight("input_layernorm"))
        q = project(x, "self_attn.q_proj").view(batch, length, Q_HEADS, HEAD_DIM)
        k = project(x, "self_attn.k_proj").view(batch, length, KV_HEADS, HEAD_DIM)
        v = project(x, "self_attn.v_proj").view(batch, length, KV_HEADS, HEAD_DIM)
        q = rotary(rms_norm(q, weight("self_attn.q_norm")).transpose(1, 2), cos, sin)
        k = rotary(rms_norm(k, weight("self_attn.k_norm")).transpose(1, 2), cos, sin)
        ratio = Q_HEADS // KV_HEADS
        k = k.repeat_interleave(ratio, dim=1)
        v = v.transpose(1, 2).repeat_interleave(ratio, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        hidden = hidden + project(attn.transpose(1, 2).reshape(batch, length, -1), "self_attn.o_proj")
        gate_input = rms_norm(hidden, weight("post_attention_layernorm"))
        down_input = F.silu(project(gate_input, "mlp.gate_proj")) * project(gate_input, "mlp.up_proj")
        if layer == LAYER:
            attention_data = split(x)
            return {
                "q_proj": attention_data,
                "v_proj": attention_data,
                "gate_proj": split(gate_input),
                "down_proj": split(down_input),
            }
        hidden = hidden + project(down_input, "mlp.down_proj")
    raise RuntimeError("activation capture did not reach the requested layer")


def quality(actual, target):
    actual = actual.float()
    magnitude = target.square().sum()
    return {
        "cosine_mean": F.cosine_similarity(actual, target, dim=-1).mean().item(),
        "relative_rms_error": ((actual - target).square().sum() / magnitude).sqrt().item(),
    }


@torch.no_grad()
def graph_time(operator, x, *, calls_per_graph=8, replays=6, samples=5):
    """Wall time includes graph submission and completion, amortised over the captured calls."""
    stream = torch.cuda.Stream()
    caller = torch.cuda.current_stream()
    stream.wait_stream(caller)
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = operator(x)
    caller.wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(calls_per_graph):
            output = operator(x)
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()

    event_times, wall_times = [], []
    operations = replays * calls_per_graph
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        begin = time.perf_counter_ns()
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        wall_times.append((time.perf_counter_ns() - begin) / (1000 * operations))
        event_times.append(start.elapsed_time(end) * 1000 / operations)
    del output
    return {
        "graph_wall_us_median": statistics.median(wall_times),
        "graph_event_us_median": statistics.median(event_times),
        "graph_calls_per_capture": calls_per_graph,
        "graph_replays_per_sample": replays,
        "graph_samples": samples,
    }


class PrunedFFT:
    """Sparse complex coefficients of the full orthonormal 2-D DFT."""

    def __init__(self, spectrum, keep):
        self.rows, self.cols = spectrum.shape
        flat = spectrum.flatten()
        ids = torch.arange(flat.numel(), device=flat.device)
        # A real matrix has conjugate pairs and four self-conjugate frequencies.
        partner = (-(ids // self.cols) % self.rows) * self.cols + (-(ids % self.cols) % self.cols)
        pairs = ids[ids < partner]
        singletons = ids[ids == partner]
        energy = flat.abs().square()
        pair_energy = energy[pairs] + energy[partner[pairs]]
        single_energy, single_order = energy[singletons].sort(descending=True)
        pair_scores, pair_order = pair_energy.topk(keep // 2, sorted=True)
        # Exact coefficient count with maximum retained energy subject to real symmetry.
        options = range(keep % 2, min(keep, singletons.numel()) + 1, 2)
        options = [n for n in options if (keep - n) // 2 <= pairs.numel()]
        totals = torch.stack([
            pair_scores[:(keep - n) // 2].sum() + single_energy[:n].sum() for n in options
        ])
        n_single = options[totals.argmax().item()]
        selected = pairs[pair_order[:(keep - n_single) // 2]]
        kept = torch.cat((selected, partner[selected], singletons[single_order[:n_single]])).sort().values
        self.output_ids = kept // self.cols
        self.input_ids = kept % self.cols
        self.coefficients = flat[kept].contiguous()
        self.energy_fraction = (energy[kept].sum() / energy.sum()).item()

    def forward(self, x):
        # If F = FFT2(W), then row-batched y = IFFT(IFFT(x) @ F.T).
        xf = torch.fft.ifft(x.float(), dim=-1, norm="ortho")
        products = xf.index_select(-1, self.input_ids) * self.coefficients
        reduced = xf.new_zeros((x.shape[0], self.rows))
        reduced.index_add_(1, self.output_ids, products)
        return torch.fft.ifft(reduced, dim=-1, norm="ortho").real.contiguous()

    def dense_reference(self):
        kept = self.coefficients.new_zeros((self.rows, self.cols))
        kept[self.output_ids, self.input_ids] = self.coefficients
        return torch.fft.ifft2(kept, norm="ortho").real.contiguous()


def fwht(x):
    width = x.shape[-1]
    shape = x.shape
    for stride in (1, 2, 4, 8, 16):
        lhs, rhs = x.reshape(*shape[:-1], width // (2 * stride), 2, stride).unbind(-2)
        x = torch.stack((lhs + rhs, lhs - rhs), dim=-2).reshape(shape)
    return x * (1 / math.sqrt(width))


class Monarch(torch.nn.Module):
    """Two block-diagonal factors with a fixed cross-block signed Hadamard lift."""

    def __init__(self, rows, cols, device):
        super().__init__()
        if cols % BLOCKS or rows % BLOCKS:
            raise ValueError("projection dimensions must be divisible by the Monarch block count")
        self.input_width = cols // BLOCKS
        self.left = torch.nn.Parameter(
            torch.randn(BLOCKS, self.input_width, BLOCKS, device=device) / math.sqrt(self.input_width)
        )
        self.right = torch.nn.Parameter(
            torch.randn(BLOCKS, BLOCKS, rows // BLOCKS, device=device) / math.sqrt(BLOCKS)
        )
        # Nonseparable signs between the two axes keep the lift from collapsing
        # into independent per-block basis changes.
        sign = torch.randint(0, 2, (BLOCKS, BLOCKS), device=device, dtype=torch.int32)
        self.register_buffer("sign", (2 * sign - 1).float())
        self.register_buffer("hadamard", fwht(torch.eye(BLOCKS, device=device)))

    def forward(self, x, left, right, sign, butterfly):
        batch = x.shape[0]
        x = x.reshape(batch, BLOCKS, self.input_width).transpose(0, 1)
        x = torch.bmm(x, left).permute(1, 2, 0).contiguous()
        x = fwht(x) if butterfly else x @ self.hadamard
        x = (x * sign).transpose(1, 2).contiguous()
        x = fwht(x) if butterfly else x @ self.hadamard
        x = torch.bmm(x.permute(2, 0, 1).contiguous(), right)
        return x.transpose(0, 1).reshape(batch, -1)

    @torch.no_grad()
    def fp16_operator(self, x_scale, y_scale):
        left = (self.left / x_scale).half()
        right = (self.right * y_scale).half()
        sign = self.sign.half()
        return lambda x: self.forward(x, left, right, sign, butterfly=True)

    def fit(self, x, target):
        optimiser = torch.optim.Adam(self.parameters(), lr=0.015, fused=True)
        for step in range(FIT_STEPS):
            if step % math.ceil(len(x) / FIT_BATCH) == 0:
                order = torch.randperm(len(x), device=x.device)
            offset = (step % math.ceil(len(x) / FIT_BATCH)) * FIT_BATCH
            ids = order[offset:offset + FIT_BATCH]
            predicted = self.forward(x[ids], self.left, self.right, self.sign, butterfly=False)
            loss = F.mse_loss(predicted, target[ids])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()


class SignSVD:
    """Packed sign/128-weight FP16 scale plus a budgeted FP16 low-rank residual."""

    def __init__(self, weight):
        rows, cols = weight.shape
        n = weight.numel()
        budget_bits = 5 * n // 2
        if n % 2 or cols % GROUP:
            raise ValueError("projection cannot be represented by this sign group layout")
        grouped = weight.reshape(rows, cols // GROUP, GROUP)
        self.scale = grouped.abs().mean(dim=-1).half()
        bits = (grouped >= 0).to(torch.uint8).reshape(rows, cols // 8, 8)
        self.shifts = torch.arange(8, device=weight.device, dtype=torch.uint8)
        self.packed = ((bits << self.shifts).sum(dim=-1, dtype=torch.int32)).to(torch.uint8)
        residual = weight - self.sign_weight(torch.float32)
        remaining = budget_bits - 8 * self.packed.numel() - 16 * self.scale.numel()
        half_slots, extra_bits = divmod(remaining, 16)
        self.rank, tail = divmod(half_slots, rows + cols)
        if extra_bits or (tail and tail <= min(rows, cols)):
            raise ValueError("budget has insufficient space for an exact partial residual component")
        self.tail = None
        count = min(rows, cols, self.rank + bool(tail) + 8)
        u, sigma, v = torch.svd_lowrank(residual, q=count, niter=2)
        root = sigma.sqrt()
        uf, vf = u * root, v * root
        self.u = uf[:, :self.rank].to(dtype=torch.float16, memory_format=torch.contiguous_format)
        self.v = vf[:, :self.rank].T.to(dtype=torch.float16, memory_format=torch.contiguous_format)
        factors = [self.u, self.v]
        if tail:
            # Spend the exact remainder on the next singular component,
            # placing the truncated factor on a fixed prefix of its axis.
            self.tail_rows = rows >= cols
            u_count = tail - cols if self.tail_rows else rows
            v_count = cols if self.tail_rows else tail - rows
            self.tail = (
                uf[:u_count, self.rank:self.rank + 1].to(dtype=torch.float16, memory_format=torch.contiguous_format),
                vf[:v_count, self.rank:self.rank + 1].T.to(dtype=torch.float16, memory_format=torch.contiguous_format),
            )
            factors.extend(self.tail)
        self.storage_bytes = sum(t.numel() * t.element_size() for t in [self.packed, self.scale, *factors])
        if self.storage_bytes * 8 != budget_bits:
            raise RuntimeError("sign and residual payload does not meet the exact 2.5 bpw budget")

    def sign_weight(self, dtype):
        bits = ((self.packed.unsqueeze(-1) >> self.shifts) & 1).to(dtype)
        groups = bits.reshape(*self.scale.shape, GROUP) * 2 - 1
        return (groups * self.scale.unsqueeze(-1)).flatten(1)

    def forward(self, x):
        y = F.linear(x, self.sign_weight(x.dtype))
        y = y + F.linear(F.linear(x, self.v), self.u)
        if self.tail is not None:
            u, v = self.tail
            if self.tail_rows:
                part = F.linear(F.linear(x, v), u)
                y = y + F.pad(part, (0, y.shape[-1] - part.shape[-1]))
            else:
                y = y + F.linear(F.linear(x[:, :v.shape[-1]], v), u)
        return y


def benchmark_projection(weight, activations):
    train, held = activations
    rows, cols = weight.shape
    target = F.linear(held, weight)
    held16 = held.half()
    one = held16[:1].contiguous()
    w16 = weight.half()

    def dense(x):
        return F.linear(x, w16)

    report = {
        "weight_shape": [rows, cols],
        "calibration_tokens": len(train),
        "held_out_tokens": len(held),
        "dense_fp16": {**quality(dense(held16), target), **graph_time(dense, one)},
    }
    spectrum = torch.fft.fft2(weight, norm="ortho")
    for denominator, label in ((4, "25pct"), (8, "12_5pct")):
        keep = weight.numel() // denominator
        fft = PrunedFFT(spectrum, keep)
        dense_pruned = fft.dense_reference()
        check_input = held16[:4]
        check = F.linear(check_input.float(), dense_pruned)
        fft_path_error = quality(fft.forward(check_input), check)["relative_rms_error"]
        if not fft_path_error < 2e-4:
            raise RuntimeError(f"sparse FFT path disagrees with inverse-FFT reference: {fft_path_error}")
        report[f"fft_{label}"] = {
            **quality(F.linear(held16.float(), dense_pruned), target),
            **graph_time(fft.forward, one),
            "coefficients_kept": keep,
            "spectral_energy_retained_fraction": fft.energy_fraction,
            "sparse_runtime_reference_relative_rms_error": fft_path_error,
        }

    training_target = F.linear(train, weight)
    x_scale = train.square().mean().sqrt().clamp_min(1e-7)
    y_scale = training_target.square().mean().sqrt().clamp_min(1e-7)
    monarch = Monarch(rows, cols, weight.device)
    fresh = monarch.fp16_operator(x_scale, y_scale)
    initial_timing = graph_time(fresh, one)
    initial_quality = quality(fresh(held16), target)
    monarch.fit(train / x_scale, training_target / y_scale)
    fitted = monarch.fp16_operator(x_scale, y_scale)
    report["monarch_fwht"] = {
        **quality(fitted(held16), target),
        **graph_time(fitted, one),
        "fresh_initialisation": {**initial_quality, **initial_timing},
        "blocks_per_factor": BLOCKS,
        "factor_parameters": BLOCKS * (rows + cols),
        "adam_steps": FIT_STEPS,
        "adam_max_minibatch_tokens": FIT_BATCH,
    }

    with torch.no_grad():
        sign = SignSVD(weight)
        report["sign_svd"] = {
            **quality(sign.forward(held16), target),
            **graph_time(sign.forward, one),
            "packed_payload_bytes": sign.storage_bytes,
            "bits_per_weight": 8 * sign.storage_bytes / weight.numel(),
            "group_size": GROUP,
            "fp16_residual_rank": sign.rank,
            "partial_rank_one_factor_lengths": (
                [part.numel() for part in sign.tail] if sign.tail is not None else None
            ),
        }
    dense_wall = report["dense_fp16"]["graph_wall_us_median"]
    for mode in ("fft_25pct", "fft_12_5pct", "monarch_fwht", "sign_svd"):
        report[mode]["speedup_over_dense_graph_wall"] = dense_wall / report[mode]["graph_wall_us_median"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("track4_spectral.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    with SafeTensorReader(args.checkpoint) as reader:
        device = torch.cuda.get_device_properties(0)
        torch.cuda.reset_peak_memory_stats()
        activations = capture_activations(reader)
        results = {}
        for name, path in PROJECTIONS.items():
            weight = reader.tensor(f"model.layers.{LAYER}.{path}.weight", torch.float32)
            results[name] = benchmark_projection(weight, activations[name])

    summary = {
        "track": 4,
        "checkpoint": str(args.checkpoint),
        "layer": LAYER,
        "seed": SEED,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu": {
            "name": device.name,
            "compute_capability": [device.major, device.minor],
            "total_memory_bytes": device.total_memory,
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "activation_source": "causal BF16 Qwen3 forward on separately written, embedded tokenised passages",
        "calibration_documents": len(CALIBRATION_IDS),
        "held_out_documents": len(HELD_OUT_IDS),
        "quality_reference": "FP32 projection on unrounded held-out activations",
        "quality_cosine_reduction": "mean of per-token cosines",
        "graph_batch_tokens": 1,
        "fft_runtime": "FP32 complex 1-D IFFTs with sparse gather and index-add",
        "fft_quality_path": "inverse-FFT dense reference on FP16 inputs promoted to FP32",
        "sign_svd_runtime": "unpack and scale sign payload on replay, then sign and residual GEMMs",
        "sign_svd_budget_scope": "persistent weight payload, excluding shared bit-shift constants and temporaries",
        "monarch_runtime": "FP16 grouped matmuls and explicit Hadamard butterflies",
        "projections": results,
        "elapsed_wall_seconds": time.perf_counter() - start,
    }
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as file:
            file.write(payload)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(payload, end="")


if __name__ == "__main__":
    main()
