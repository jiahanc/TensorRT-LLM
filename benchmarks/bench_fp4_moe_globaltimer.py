# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark FP4 MoE autotune with globaltimer-based profiling.

Uses the RoutingRenormalize_large_experts config (2048 experts, top_k=32)
and records kernel duration via the GPU globaltimer register instead of
CUDA events, which is more reliable under confidential compute.

Usage:
    python bench_fp4_moe_globaltimer.py --num-tokens 1024 --warmup 3 --repeat 10
"""

import argparse
import os
import sys
from enum import Enum

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.modules.fused_moe import RoutingMethodType
from tensorrt_llm._torch.utils import next_positive_power_of_2
from tensorrt_llm.bindings.internal.runtime import record_global_timer
from tensorrt_llm.quantization.utils.fp4_utils import (
    reorder_rows_for_gated_act_gemm, shuffle_matrix_a, shuffle_matrix_sf_a)


class ActType(Enum):
    SwiGlu = 0
    Relu2 = 1
    Silu = 2


NS_PER_MS = 1e6


def quant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16
    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a.cuda(), a_global_sf.cuda(),
                                                 sf_vec_size, use_ue8m0,
                                                 is_sf_swizzled_layout)
    return a_fp4, a_sf, a_global_sf


def quant_fp4_batches(a,
                      num_experts,
                      use_ue8m0=False,
                      is_sf_swizzled_layout=True):
    quant_a = []
    sfs = []
    global_sfs = []
    for i in range(num_experts):
        a_fp4, a_sf, a_global_sf = quant_fp4(a[i], use_ue8m0,
                                             is_sf_swizzled_layout)
        quant_a.append(a_fp4)
        sfs.append(a_sf)
        global_sfs.append(a_global_sf)
    result_a_quant = torch.stack(quant_a)
    result_a_scales = torch.stack(sfs)
    result_global_sfs = torch.stack(global_sfs)
    return result_a_quant, result_a_scales, result_global_sfs


def quant_dequant_fp4(a, use_ue8m0=False, is_sf_swizzled_layout=True):
    a_global_sf = (448 * 6) / a.float().abs().nan_to_num().max()
    sf_vec_size = 16
    a_fp4, a_sf = torch.ops.trtllm.fp4_quantize(a.cuda(), a_global_sf.cuda(),
                                                 sf_vec_size, use_ue8m0,
                                                 is_sf_swizzled_layout)
    float_tensor = torch.ops.tensorrt_llm.e2m1_and_ufp8sf_scale_to_float_v2(
        a_fp4.cpu(),
        a_sf.cpu().reshape(-1),
        (1 / a_global_sf).cpu(), sf_vec_size, 1, is_sf_swizzled_layout)
    return float_tensor.cuda(), a_global_sf


def routing_reference(expertLogits, topK, padding):
    originalDevice = expertLogits.device
    expertLogits = expertLogits.cpu()
    numTokens, numExperts = expertLogits.shape
    assert topK <= numExperts

    numTokensPerExpert = torch.zeros(numExperts, dtype=torch.int64)
    expandedTokenIdxToExpert = -torch.ones(numTokens * topK, dtype=torch.int64)
    expandedTokenIdxToIdxInExpert = -torch.ones(numTokens * topK,
                                                dtype=torch.int64)

    topKLogits, topKIndices = torch.topk(expertLogits, topK, dim=1)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expertIndex = topKIndices[tokenIdx, k]
            expandedTokenIdxToExpert[expandedIdx] = expertIndex
            expandedTokenIdxToIdxInExpert[expandedIdx] = numTokensPerExpert[
                expertIndex]
            numTokensPerExpert[expertIndex] += 1

    paddedTokensPerExpertPrefixSum = torch.zeros(numExperts + 1,
                                                 dtype=torch.int64)
    for ii in range(numExperts):

        def divUpMul(a, b):
            return (a + b - 1) // b * b

        paddedTokensPerExpertPrefixSum[
            ii + 1] = paddedTokensPerExpertPrefixSum[ii] + divUpMul(
                numTokensPerExpert[ii], padding)
    permutedBufferSize = paddedTokensPerExpertPrefixSum[numExperts]

    expandedTokenIdxToPermutedIdx = -torch.ones(numTokens * topK,
                                                dtype=torch.int64)
    permutedIdxToExpandedIdx = -torch.ones(permutedBufferSize,
                                           dtype=torch.int64)
    permutedIdxToTokenIdx = -torch.ones(permutedBufferSize, dtype=torch.int64)
    for tokenIdx in range(numTokens):
        for k in range(topK):
            expandedIdx = tokenIdx * topK + k
            expert = expandedTokenIdxToExpert[expandedIdx]
            offsetWithinExpert = expandedTokenIdxToIdxInExpert[expandedIdx]
            offsetForExpert = paddedTokensPerExpertPrefixSum[expert]
            permutedIdx = offsetForExpert + offsetWithinExpert

            expandedTokenIdxToPermutedIdx[expandedIdx] = permutedIdx
            permutedIdxToExpandedIdx[permutedIdx] = expandedIdx
            permutedIdxToTokenIdx[permutedIdx] = tokenIdx
    return {
        "paddedTokensPerExpertPrefixSum":
        paddedTokensPerExpertPrefixSum.to(originalDevice),
        "permutedBufferSize":
        permutedBufferSize.item(),
        "expandedTokenIdxToPermutedIdx":
        expandedTokenIdxToPermutedIdx.to(originalDevice),
        "permutedIdxToExpandedIdx":
        permutedIdxToExpandedIdx.to(originalDevice),
        "numTokensPerExpert":
        numTokensPerExpert.to(originalDevice),
        "expandedTokenIdxToExpert":
        expandedTokenIdxToExpert.to(originalDevice),
        "topKLogits":
        topKLogits.to(originalDevice),
        "permutedIdxToTokenIdx":
        permutedIdxToTokenIdx.to(originalDevice),
        "topKIndices":
        topKIndices.to(originalDevice)
    }


def routing_reference_renormalize(expert_logits, top_k, padding):
    topk_values, topk_idx = torch.topk(expert_logits, k=top_k, dim=-1)
    topk_values = torch.nn.functional.softmax(topk_values.float(), dim=-1)

    new_mask = torch.zeros_like(expert_logits)
    new_mask.scatter_(-1, topk_idx, 1)
    scores = expert_logits * new_mask

    for i in range(topk_idx.shape[0]):
        for j in range(topk_idx.shape[1]):
            scores[i, topk_idx[i, j]] = topk_values[i, j]
    permute_info = routing_reference(scores, top_k, padding)
    return permute_info, scores


def prepare_fp4_moe_data(num_tokens, hidden_size, intermediate_size,
                         num_experts, top_k, act_type, tp_size=1, ep_size=1):
    """Generate quantized FP4 MoE data and prepare shuffled weights.

    Args:
        tp_size: Tensor parallelism size. Shards intermediate_size across TP ranks.
        ep_size: Expert parallelism size. Distributes experts across EP ranks.
                 Weights are created only for local experts (num_experts // ep_size).
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    assert num_experts % ep_size == 0, \
        f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
    assert intermediate_size % tp_size == 0, \
        f"intermediate_size ({intermediate_size}) must be divisible by tp_size ({tp_size})"

    local_num_experts = num_experts // ep_size
    intermediate_size_per_partition = intermediate_size // tp_size

    routing_method_type = RoutingMethodType.Renormalize
    tile_tokens_dim = (num_tokens * top_k) // num_experts
    tile_tokens_dim = next_positive_power_of_2(tile_tokens_dim)
    tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)
    padding = tile_tokens_dim

    assert padding < 256, "Routing kernel requires that padding be less than 256"
    assert top_k <= num_experts
    assert top_k <= 32
    assert num_experts % 4 == 0

    expert_logits = torch.randn((num_tokens, num_experts),
                                device='cuda').to(torch.bfloat16)
    routing_bias = None

    intermediate_size_factor = 2 if act_type == ActType.SwiGlu else 1

    hidden_states = 2 * torch.randn(
        (num_tokens, hidden_size), device='cuda', dtype=torch.bfloat16)
    gemm1_weights = torch.randn(
        (local_num_experts, intermediate_size_factor * intermediate_size_per_partition, hidden_size),
        device='cuda',
        dtype=torch.bfloat16)
    gemm2_weights = torch.randn((local_num_experts, hidden_size, intermediate_size_per_partition),
                                device='cuda',
                                dtype=torch.bfloat16)

    use_ue8m0 = False

    hidden_states_fp4_bytes, _, hidden_states_scale_global = quant_fp4(
        hidden_states, use_ue8m0, True)
    _, hidden_states_scale_linear_fp4_bytes, _ = quant_fp4(
        hidden_states, use_ue8m0, False)

    hidden_states_fp4 = hidden_states_fp4_bytes.reshape(
        num_tokens, hidden_size // 2)
    hidden_states_scale_linear_fp4 = hidden_states_scale_linear_fp4_bytes.view(
        torch.float8_e4m3fn)

    gemm1_weights_fp4_bytes, gemm1_scales_fp4_bytes, gemm1_scales_global = quant_fp4_batches(
        gemm1_weights, local_num_experts, use_ue8m0, True)
    _, gemm1_scales_linear_fp4_bytes, _ = quant_fp4_batches(
        gemm1_weights, local_num_experts, use_ue8m0, False)

    gemm1_weights_fp4 = gemm1_weights_fp4_bytes.view(
        torch.float8_e4m3fn).reshape(local_num_experts,
                                     intermediate_size_factor *
                                     intermediate_size_per_partition,
                                     hidden_size // 2)
    gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn).reshape(local_num_experts,
                                     intermediate_size_factor *
                                     intermediate_size_per_partition,
                                     hidden_size // 16)

    gemm2_weights_fp4_bytes, gemm2_scales_fp4_bytes, gemm2_scales_global = quant_fp4_batches(
        gemm2_weights, local_num_experts, use_ue8m0, True)
    _, gemm2_scales_linear_fp4_bytes, _ = quant_fp4_batches(
        gemm2_weights, local_num_experts, use_ue8m0, False)

    gemm2_weights_fp4 = gemm2_weights_fp4_bytes.view(
        torch.float8_e4m3fn).reshape(local_num_experts, hidden_size,
                                     intermediate_size_per_partition // 2)
    gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
        torch.float8_e4m3fn).reshape(local_num_experts, hidden_size,
                                     intermediate_size_per_partition // 16)

    permute_info, scores = routing_reference_renormalize(
        expert_logits, top_k, padding)

    # Compute c_global_sf via quant-dequant round-trip on a dummy activation
    dummy_act = torch.randn(1, intermediate_size_per_partition, device='cuda',
                            dtype=torch.bfloat16)
    _, c_global_sf = quant_dequant_fp4(dummy_act, False, True)

    epilogue_tile_m = 128

    # Reorder rows of W1 for fused gated activation
    if act_type == ActType.SwiGlu:
        gemm1_weights_fp4_interleaved = []
        gemm1_scales_fp4_interleaved = []
        for i in range(local_num_experts):
            gemm1_weights_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(gemm1_weights_fp4[i].clone()))
            gemm1_scales_fp4_interleaved.append(
                reorder_rows_for_gated_act_gemm(
                    gemm1_scales_linear_fp4[i].clone()))
        gemm1_weights_fp4_interleaved = torch.stack(
            gemm1_weights_fp4_interleaved).reshape(
                local_num_experts,
                intermediate_size_factor * intermediate_size_per_partition,
                hidden_size // 2)
        gemm1_scales_fp4_interleaved = torch.stack(
            gemm1_scales_fp4_interleaved).reshape(
                local_num_experts,
                intermediate_size_factor * intermediate_size_per_partition,
                hidden_size // 16)
    else:
        gemm1_weights_fp4_interleaved = gemm1_weights_fp4.clone()
        gemm1_scales_fp4_interleaved = gemm1_scales_linear_fp4.clone()

    # Shuffle weights and scaling factors for transposed MMA output
    gemm1_weights_fp4_shuffled = []
    gemm1_scales_fp4_shuffled = []
    gemm2_weights_fp4_shuffled = []
    gemm2_scales_fp4_shuffled = []
    for i in range(local_num_experts):
        gemm1_weights_fp4_shuffled.append(
            shuffle_matrix_a(
                gemm1_weights_fp4_interleaved[i].view(torch.uint8),
                epilogue_tile_m))
        gemm1_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(
                gemm1_scales_fp4_interleaved[i].view(torch.uint8),
                epilogue_tile_m))
        gemm2_weights_fp4_shuffled.append(
            shuffle_matrix_a(gemm2_weights_fp4[i].view(torch.uint8),
                             epilogue_tile_m))
        gemm2_scales_fp4_shuffled.append(
            shuffle_matrix_sf_a(
                gemm2_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m))

    gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
    gemm1_scales_fp4_shuffled = torch.stack(gemm1_scales_fp4_shuffled).view(
        torch.float8_e4m3fn).reshape(local_num_experts,
                                     intermediate_size_factor *
                                     intermediate_size_per_partition,
                                     hidden_size // 16)
    gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
    gemm2_scales_fp4_shuffled = torch.stack(gemm2_scales_fp4_shuffled).view(
        torch.float8_e4m3fn).reshape(local_num_experts, hidden_size,
                                     intermediate_size_per_partition // 16)

    if act_type == ActType.SwiGlu:
        scale_c_fc1 = c_global_sf * (1.0 / gemm1_scales_global) * (
            1.0 / hidden_states_scale_global)
    else:
        scale_c_fc1 = torch.full_like(gemm1_scales_global, c_global_sf)

    scale_gate_fc1 = (1.0 / gemm1_scales_global) * (
        1.0 / hidden_states_scale_global)
    scale_c_fc2 = (1.0 / c_global_sf) * (1.0 / gemm2_scales_global)

    return {
        "expert_logits": expert_logits,
        "routing_bias": routing_bias,
        "hidden_states_fp4": hidden_states_fp4,
        "hidden_states_scale_linear_fp4": hidden_states_scale_linear_fp4,
        "gemm1_weights_fp4_shuffled": gemm1_weights_fp4_shuffled,
        "gemm1_scales_fp4_shuffled": gemm1_scales_fp4_shuffled,
        "gemm2_weights_fp4_shuffled": gemm2_weights_fp4_shuffled,
        "gemm2_scales_fp4_shuffled": gemm2_scales_fp4_shuffled,
        "scale_c_fc1": scale_c_fc1,
        "scale_gate_fc1": scale_gate_fc1,
        "scale_c_fc2": scale_c_fc2,
        "num_experts": num_experts,
        "local_num_experts": local_num_experts,
        "top_k": top_k,
        "intermediate_size": intermediate_size_per_partition,
        "routing_method_type": routing_method_type,
        "act_type": act_type,
    }


def record_globaltimer_pair(stream):
    """Create start/end timestamp tensors and return record/elapsed helpers."""
    start_ts = torch.empty(1, dtype=torch.int64, device='cuda')
    end_ts = torch.empty(1, dtype=torch.int64, device='cuda')

    def record_start():
        record_global_timer(start_ts.data_ptr(), stream)

    def record_end():
        record_global_timer(end_ts.data_ptr(), stream)

    def elapsed_ms():
        return (end_ts.item() - start_ts.item()) / NS_PER_MS

    return record_start, record_end, elapsed_ms


def run_benchmark(data, warmup, repeat, use_autotune=True):
    """Run the FP4 MoE kernel with globaltimer profiling inside a CUDA graph.

    Args:
        use_autotune: If True, run autotuner during warmup to select the best
                      tactic. If False, use the default tactic without tuning.
    """
    d = data
    act_type_val = d["act_type"].value
    stream = torch.cuda.current_stream()

    def run_kernel():
        return torch.ops.trtllm.fp4_block_scale_moe_runner(
            d["expert_logits"],
            d["routing_bias"],
            d["hidden_states_fp4"],
            d["hidden_states_scale_linear_fp4"],
            d["gemm1_weights_fp4_shuffled"],
            d["gemm1_scales_fp4_shuffled"],
            None,  # gemm1_bias
            None,  # swiglu_alpha
            None,  # swiglu_beta
            None,  # swiglu_limit
            d["gemm2_weights_fp4_shuffled"],
            d["gemm2_scales_fp4_shuffled"],
            None,  # gemm2_bias
            d["scale_c_fc1"],
            d["scale_gate_fc1"],
            d["scale_c_fc2"],
            d["num_experts"],
            d["top_k"],
            None,  # n_groups
            None,  # top_k_groups
            d["intermediate_size"],
            0,  # local_expert_offset (GPU 0)
            d["local_num_experts"],
            None,  # routed_scaling
            d["routing_method_type"],
            do_finalize=True,
            topk_ids=None,
            topk_weights=None,
            act_type=act_type_val)

    tune_label = "with autotune" if use_autotune else "without autotune"

    # Warmup (eager, no graph)
    print(f"Running warmup ({warmup} iterations, {tune_label})...")
    AutoTuner.get().clear_cache()
    with autotune(use_autotune):
        for _ in range(warmup):
            run_kernel()
    torch.cuda.synchronize()
    print("Warmup complete.")

    # Capture the kernel into a CUDA graph
    print("Capturing CUDA graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with autotune(use_autotune):
            with torch.cuda.graph(graph):
                run_kernel()
    torch.cuda.synchronize()
    print("CUDA graph captured.")

    # Start CUDA profiler after warmup so traces only capture timed runs
    torch.cuda.cudart().cudaProfilerStart()

    # Timed runs: globaltimer outside graph, replay inside
    print(f"Profiling with globaltimer ({repeat} iterations, CUDA graph, "
          f"{tune_label})...")
    times_ms = []
    record_start, record_end, elapsed_ms = record_globaltimer_pair(stream)

    for i in range(repeat):
        torch.cuda.synchronize()
        record_start()

        graph.replay()

        record_end()
        torch.cuda.synchronize()

        t = elapsed_ms()
        times_ms.append(t)
        print(f"  iter {i}: {t:.4f} ms")

    torch.cuda.cudart().cudaProfilerStop()

    return times_ms


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP4 MoE autotune with globaltimer profiling")
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--act-type",
                        type=str,
                        default="Silu",
                        choices=["SwiGlu", "Relu2","Silu"],
                        help="Activation type for the MoE kernel")
    parser.add_argument("--tp-size",
                        type=int,
                        default=4,
                        help="Tensor parallelism size (shards intermediate_size)")
    parser.add_argument("--ep-size",
                        type=int,
                        default=1,
                        help="Expert parallelism size (distributes experts across ranks)")
    parser.add_argument("--no-autotune",
                        action="store_true",
                        help="Disable autotuner, use default tactic")
    args = parser.parse_args()

    # RoutingRenormalize_large_experts config
    num_experts = 1536
    top_k = 32
    act_type = ActType[args.act_type]

    local_num_experts = num_experts // args.ep_size
    intermediate_size_per_partition = args.intermediate_size // args.tp_size

    print("=" * 60)
    print("FP4 MoE Benchmark — globaltimer profiling")
    print("=" * 60)
    print(f"  Config: RoutingRenormalize_large_experts")
    print(f"  num_experts={num_experts}, top_k={top_k}")
    print(f"  num_tokens={args.num_tokens}, hidden_size={args.hidden_size}")
    print(f"  intermediate_size={args.intermediate_size}")
    print(f"  tp_size={args.tp_size}, ep_size={args.ep_size}")
    print(f"  local_num_experts={local_num_experts}, "
          f"intermediate_size_per_partition={intermediate_size_per_partition}")
    print(f"  act_type={act_type.name}")
    print(f"  autotune={'OFF' if args.no_autotune else 'ON'}")
    print(f"  warmup={args.warmup}, repeat={args.repeat}")
    print("=" * 60)

    print("\nPreparing quantized FP4 data...")
    data = prepare_fp4_moe_data(args.num_tokens, args.hidden_size,
                                args.intermediate_size, num_experts, top_k,
                                act_type, tp_size=args.tp_size,
                                ep_size=args.ep_size)
    print("Data preparation complete.\n")

    use_autotune = not args.no_autotune
    times_ms = run_benchmark(data, args.warmup, args.repeat,
                             use_autotune=use_autotune)

    avg = sum(times_ms) / len(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    median = (sorted_times[n // 2] if n % 2 == 1
              else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2)
    print(f"\n{'=' * 60}")
    print(f"Results ({n} iterations):")
    print(f"  avg    = {avg:.4f} ms")
    print(f"  median = {median:.4f} ms")
    print(f"  min    = {mn:.4f} ms")
    print(f"  max    = {mx:.4f} ms")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
