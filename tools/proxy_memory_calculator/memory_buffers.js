/* SPDX-License-Identifier: Apache-2.0
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * CSA MFSDP v2 buffer padding, in bytes; no training or browser dependencies.
 * Sources in this checkout (1cffd9e1655a):
 *   megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/
 *     layout_builder.py:39 (packing), module.py:548 (dtype groups),
 *     parameter_group.py:277 (persistent buffers), muon.py:213 (step buffers).
 *   megatron/core/distributed/fsdp/mcore_adapter_v2.py:186 (wrapping units).
 * Shape sources: transformer/experimental_attention_variant/{csa.py,
 * deepseek_v4_hybrid_attention.py}, transformer/multi_token_prediction.py,
 * models/gpt/gpt_model.py, relative to megatron/core/.
 */
(function (root) {
    'use strict';
    const product = xs => xs.reduce((a, b) => a * b, 1);
    const roundUp = (n, multiple) => Math.ceil(n / multiple) * multiple;
    function safe(n) {
        if (!Number.isSafeInteger(n) || n < 0) throw Error('Buffer size exceeds safe integer precision');
        return n;
    }
    function gcd(a, b) {
        while (b) [a, b] = [b, a % b];
        return a;
    }

    // Size-equivalent translation of build_for_row_atomic, including conjugate
    // tensors and aligned fragments. A single ceil(sum(numel)) misses internal gaps.
    function pack(shapes, meshSize, blockSize, padded) {
        const tensors = shapes.map(shape => {
            if (!shape.length || shape.some(x => !Number.isSafeInteger(x) || x < 1)) {
                throw Error('Buffer shapes must contain positive integer dimensions');
            }
            if (blockSize === 32 && (shape.length !== 2 || shape[1] % 32)) {
                throw Error('MXFP8 weights require 2D shapes with dimensions divisible by 32');
            }
            if (shape[0] % blockSize) throw Error('MXFP8 weights require complete 32-row blocks');
            return {size: safe(product(shape)), align: safe(blockSize * product(shape.slice(1)))};
        });
        const logical = safe(tensors.reduce((n, t) => n + t.size, 0));
        if (!padded) return {logical, allocated: logical, alignment: null, internalGap: 0, tailPadding: 0};
        const alignment = tensors.reduce((n, t) => safe(n / gcd(n, t.align) * t.align), 1);
        const regular = tensors.filter(t => t.size >= alignment);
        const fragments = tensors.filter(t => t.size < alignment).sort((a, b) => b.size - a.size);
        let end = 0;
        while (regular.length) {
            const tensor = regular.shift(), remainder = tensor.size % alignment;
            if (!remainder) { end += tensor.size; continue; }
            let gap = end + tensor.size;
            end += roundUp(tensor.size, alignment);
            let gapEnd = end;
            const mate = regular.findIndex(t => t.size % alignment && remainder + t.size % alignment <= alignment);
            if (mate !== -1) {
                const other = regular.splice(mate, 1)[0];
                gapEnd = end - other.size % alignment;
                end += Math.floor(other.size / alignment) * alignment;
            }
            for (let i = 0; i < fragments.length;) {
                const fragment = fragments[i], start = roundUp(gap, fragment.align);
                if (start + fragment.size <= gapEnd) {
                    gap = start + fragment.size;
                    fragments.splice(i, 1);
                } else i++;
            }
        }
        for (const t of fragments) end = roundUp(end, t.align) + t.size;
        const allocated = safe(roundUp(end, safe(alignment * meshSize)));
        return {logical, allocated, alignment, internalGap: end - logical, tailPadding: allocated - end};
    }

    // Named-parameter traversal order, with routed experts excluded from the
    // TransformerLayer unit. Plain grouped-O, compressors and router stay BF16.
    function layerParameters(c, moe) {
        const h = c.hidden, d = c.headDim, di = c.indexDim, ah = c.heads * d;
        const params = [];
        const add = (shape, dtype = 'bf16', muon = true) => params.push({shape, dtype, muon});
        const te = shape => add(shape, c.fp8Gather ? 'mxfp8' : 'bf16');
        const norm = width => add([width], 'bf16', false);
        norm(h);
        add([c.oGroups * c.oLora, ah / c.oGroups]);
        add([c.heads], 'fp32', false);
        add([c.ratio, 2 * d], 'fp32', false);
        add([2 * d, h]); add([2 * d, h]); norm(d);
        te([c.indexHeads * di, c.qLora]);
        add([c.indexHeads, h]);
        add([c.ratio, 2 * di], 'fp32', false);
        add([2 * di, h]); add([2 * di, h]); norm(di);
        te([h, c.oGroups * c.oLora]);
        te([c.qLora, h]); te([ah, c.qLora]); te([d, h]);
        norm(d); norm(c.qLora);
        // MoE has a standalone pre-MLP norm; dense TE FC1 fuses that norm.
        // Both contribute this BF16 vector to the same TransformerLayer group.
        norm(h);
        if (moe) add([c.experts, h]);
        const f = moe ? c.sharedFfn : c.ffn;
        if (f) { te([2 * f, h]); te([h, f]); }
        return params;
    }

    /**
     * chunks: one PP rank's chunk objects from CSAMemory.parseLayout, or a flat
     * list of all PP chunks. finalNorm marks the LAST DECODER chunk, not mL.
     * largestDenseGroup / largestExpertGroup are allocated ELEMENT counts,
     * including root groups; multiply by 4 for FP32 partial_grad bytes.
     * paddingBytes is ONLY the extra resident compute + master + gradient memory.
     * details lists every actual unit/dtype bucket (also when paddedShards=false).
     * No padding is added to optimizer states, which own valid parameter views.
     * TE padded-scale copies, BF16 quantization caches and collective workspaces
     * are outside this correction. Unpadded shard imbalance is not a padding cost.
     */
    function estimate(config, chunks, derived) {
        const c = {denseZeroStage: 2, paddedShards: false, ...config};
        const {domain, edp} = derived;
        for (const n of [domain, edp, c.inner]) {
            if (!Number.isSafeInteger(n) || n < 1) throw Error('Invalid MFSDP shard count');
        }
        if (domain % c.inner || c.experts % c.ep) throw Error('Nonintegral MFSDP topology');
        if (!['optim', 'no_shard'].includes(c.denseOuter)) throw Error('Unsupported dense outer strategy');
        if (![1, 2, 3].includes(c.denseZeroStage)) throw Error('Invalid dense ZeRO stage');
        if (c.ratio !== 4) throw Error('This shape model requires CSA ratio=4');
        const details = [];
        function unit(name, kind, params, chunkIndex) {
            const buckets = new Map();
            for (const p of params) {
                if (!buckets.has(p.dtype)) buckets.set(p.dtype, []);
                buckets.get(p.dtype).push(p);
            }
            for (const [dtype, ps] of buckets) {
                const meshSize = kind === 'expert' ? edp : domain;
                const optimizerShards = kind === 'expert' ? edp : (c.denseOuter === 'optim' ? domain : c.inner);
                const gradientShards = kind === 'expert' || c.denseZeroStage === 1 ? 1 : c.inner;
                const parameterShards = kind === 'dense' && c.denseZeroStage === 3 ? c.inner : 1;
                const blockSize = dtype === 'mxfp8' ? 32 : 1;
                const shapes = ps.map(p => p.shape);
                // Requested comparison convention: ZeRO-1 is an unpadded
                // baseline, including its step workspaces. This is not a claim
                // that the MFSDP runtime automatically disables padded shards.
                const layout = pack(shapes, meshSize, blockSize, c.paddedShards && c.denseZeroStage !== 1);
                const extra = layout.allocated - layout.logical;
                const computeWidth = dtype === 'mxfp8' ? 33 / 16 : (dtype === 'fp32' ? 4 : 2);
                const computePaddingBytes = extra * computeWidth / parameterShards;
                const masterPaddingBytes = 4 * extra / optimizerShards;
                const gradientPaddingBytes = 4 * extra / gradientShards;
                const hasMuon = ps.some(p => p.muon);
                details.push({name, chunkIndex, kind, dtype, shapes, meshSize, optimizerShards,
                    gradientShards, parameterShards, blockSize, alignmentElements: layout.alignment,
                    logicalElements: layout.logical, allocatedElements: layout.allocated,
                    paddingElements: extra, internalGapElements: layout.internalGap,
                    tailPaddingElements: layout.tailPadding,
                    fullComputeBytes: layout.allocated * computeWidth,
                    // Autograd produces full logical gradients before packing.
                    autogradGradientBytes: layout.logical * (dtype === 'fp32' ? 4 : 2),
                    computePaddingBytes, masterPaddingBytes, gradientPaddingBytes,
                    paddingBytes: computePaddingBytes + masterPaddingBytes + gradientPaddingBytes,
                    // Two sharded DBuffer temporaries coexist during Muon scatter.
                    // These are step transients, NOT another resident optimizer state.
                    muonWorkspaceBytes: hasMuon ? 8 * layout.allocated / optimizerShards : 0,
                    muonWorkspacePaddingBytes: hasMuon ? 8 * extra / optimizerShards : 0});
            }
        }
        for (const chunk of chunks.flat()) {
            const prefix = `chunk${chunk.index}`;
            for (const [type, count] of [['dense', chunk.dense], ['moe', chunk.moe], ['mtp', chunk.mtp]]) {
                for (let i = 0; i < count; i++) {
                    const name = `${prefix}.${type}${i}`;
                    unit(name, 'dense', layerParameters(c, type !== 'dense'), chunk.index);
                    if (type !== 'dense') {
                        const ps = [], dtype = c.fp8Gather ? 'mxfp8' : 'bf16';
                        for (const shape of [[2 * c.moeFfn, c.hidden], [c.hidden, c.moeFfn]]) {
                            for (let e = 0; e < c.experts / c.ep; e++) ps.push({shape, dtype, muon: true});
                        }
                        unit(`${name}.experts`, 'expert', ps, chunk.index);
                    }
                }
            }
            // GPT root owns everything outside its TransformerLayer/expert children.
            // MTP eh_proj and its three norms remain here even in an mL-only chunk.
            const ps = [], h = c.hidden;
            const add = (shape, dtype = 'bf16', muon = false) => ps.push({shape, dtype, muon});
            if (chunk.embedding || chunk.mtp) add([c.vocab, h]);
            if (chunk.finalNorm) add([h]);
            for (let i = 0; i < chunk.mtp; i++) {
                add([h]); add([h]);
                add([h, 2 * h], c.fp8Gather ? 'mxfp8' : 'bf16', true);
                add([h]);
            }
            if (chunk.output) add([c.vocab, h]);
            unit(`${prefix}.root`, 'dense', ps, chunk.index);
        }
        return {
            paddingBytes: details.reduce((n, x) => n + x.paddingBytes, 0),
            largestDenseGroup: Math.max(0, ...details.filter(x => x.kind === 'dense').map(x => x.allocatedElements)),
            largestExpertGroup: Math.max(0, ...details.filter(x => x.kind === 'expert').map(x => x.allocatedElements)),
            details,
        };
    }

    // module.py:_reduce_gradient_groups retains one preceding module batch while
    // staging the current one. A batch contains ALL dtype buckets of that unit.
    // Expert units still pack gradients when EDP=1 (no NCCL collective needed).
    // Include GPT roots: embedding/output can be much larger than a decoder.
    function gradientWorkspace(config, details) {
        // ZeRO-1 in this calculator is the standard DDP comparison convention.
        if (config.denseZeroStage === 1) return {bytes: 0, partialBytes: 0, autogradBytes: 0, units: [], retained: []};
        const byName = new Map();
        for (const group of details) {
            if (!byName.has(group.name)) byName.set(group.name, {
                name: group.name, kind: group.kind, chunkIndex: group.chunkIndex,
                partialBytes: 0, autogradBytes: 0,
            });
            const unit = byName.get(group.name);
            unit.partialBytes += 4 * group.allocatedElements;
            unit.autogradBytes += group.autogradGradientBytes;
        }
        const units = [...byName.values()];
        const retained = [...units].sort((a, b) => b.partialBytes - a.partialBytes).slice(0, 2);
        const partialBytes = retained.reduce((n, x) => n + x.partialBytes, 0);
        const autogradBytes = Math.max(0, ...units.map(x => x.autogradBytes));
        return {bytes: partialBytes + autogradBytes, partialBytes, autogradBytes, units, retained};
    }
    const api = {estimate, gradientWorkspace};
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    root.CSABuffers = api;
})(typeof globalThis === 'undefined' ? this : globalThis);
