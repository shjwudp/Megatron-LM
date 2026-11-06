/* SPDX-License-Identifier: Apache-2.0
 * CSA + MFSDP v2 memory estimate for proxy models. No training dependencies.
 * Shape model and limitations: memory_calculator.md. Bytes, not decimal GB.
 */
(function (root) {
    'use strict';
    const GiB = 2 ** 30;
    const buffers = typeof module !== 'undefined' && module.exports ? require('./memory_buffers.js') : root.CSABuffers;
    const sum = obj => Object.values(obj).reduce((a, b) => a + b, 0);
    const add = (dst, src, scale = 1) => {
        for (const [key, value] of Object.entries(src)) dst[key] = (dst[key] || 0) + value * scale;
        return dst;
    };
    const upstreamRecompute = ["mla_up_proj", "moe_act", "layernorm", "shared_experts", "dsa_selected_scores"];
    const supportedRecompute = ["mla_up_proj", "moe_act", "layernorm", "shared_experts", "dsa_selected_scores", "dense_fc1_act", "moe_fc1", "csa_o_proj", "csa_core_output"];
    const recomputeOrigins = Object.fromEntries(supportedRecompute.map(name =>
        [name, upstreamRecompute.includes(name) ? "main 原有" : "本分支新增"]));
    const hfsdp16 = {
    "world": 128,
    "edpMultiplier": 1,
    "pp": 4,
    "tp": 1,
    "cp": 4,
    "ep": 32,
    "inner": 8,
    "group": 4,
    "seq": 32768,
    "mbs": 1,
    "gbs": 64,
    "layers": 44,
    "denseLayers": 4,
    "hidden": 7168,
    "ffn": 20480,
    "experts": 128,
    "topk": 4,
    "moeFfn": 5120,
    "sharedFfn": 5120,
    "heads": 128,
    "headDim": 512,
    "qLora": 1536,
    "oGroups": 16,
    "oLora": 1024,
    "indexHeads": 64,
    "indexDim": 128,
    "indexTopk": 1024,
    "ratio": 4,
    "window": 128,
    "vocab": 200064,
    "layout": "Etttt|(tttt|)*10,mL",
    "fp8Gather": true,
    "epOverlap": true,
    "denseZeroStage": 2,
    "denseOuter": "optim",
    "paddedShards": false,
    "recompute": [
        "mla_up_proj",
        "moe_act",
        "layernorm",
        "shared_experts",
        "dsa_selected_scores",
        "dense_fc1_act",
        "moe_fc1",
        "csa_o_proj",
        "csa_core_output"
    ],
    "reserveGiB": 8,
    "nonTorchGiB": 20,
    "expertLoad": 1
};
    const local = {
    "world": 8,
    "edpMultiplier": 2,
    "pp": 2,
    "tp": 1,
    "cp": 2,
    "ep": 2,
    "inner": 2,
    "group": 2,
    "seq": 32768,
    "mbs": 1,
    "gbs": 16,
    "layers": 8,
    "denseLayers": 1,
    "hidden": 2048,
    "ffn": 6144,
    "experts": 64,
    "topk": 4,
    "moeFfn": 1024,
    "sharedFfn": 1024,
    "heads": 64,
    "headDim": 512,
    "qLora": 512,
    "oGroups": 8,
    "oLora": 512,
    "indexHeads": 64,
    "indexDim": 128,
    "indexTopk": 512,
    "ratio": 4,
    "window": 128,
    "vocab": 200064,
    "layout": "Ett|tt|tt|ttmL",
    "fp8Gather": true,
    "epOverlap": false,
    "denseZeroStage": 2,
    "denseOuter": "no_shard",
    "paddedShards": false,
    "recompute": [
        "mla_up_proj",
        "moe_act",
        "layernorm",
        "shared_experts",
        "dsa_selected_scores"
    ],
    "reserveGiB": 0,
    "nonTorchGiB": 0,
    "expertLoad": 1
};
    const defaults = hfsdp16;

    // Deliberately parse the small E/t/m/L layout grammar; never evaluate input code.
    function parseLayout(text, pp, denseLayers) {
        const s = text.replace(/[\s,]/g, '');
        let pos = 0;
        function sequence(nested = false) {
            let out = '';
            while (pos < s.length && s[pos] !== ')') {
                const ch = s[pos++];
                let term;
                if (ch === '(') {
                    term = sequence(true);
                    if (s[pos++] !== ')') throw Error('layout 缺少右括号');
                } else if ('EtmL|'.includes(ch)) term = ch;
                else throw Error('layout 仅支持 E/t/m/L、|、括号和 *次数');
                if (s[pos] === '*') {
                    pos++;
                    const match = s.slice(pos).match(/^\d+/);
                    if (!match || Number(match[0]) > 1000) throw Error('layout 重复次数无效');
                    pos += match[0].length;
                    term = term.repeat(Number(match[0]));
                }
                out += term;
                if (out.length > 10000) throw Error('layout 过长');
            }
            if (!nested && pos !== s.length) throw Error('layout 多余右括号');
            return out;
        }
        const expanded = sequence();
        const slots = expanded.split('|');
        if ([...expanded].filter(x=>x==='E').length!==1 || [...expanded].filter(x=>x==='m').length!==1 || [...expanded].filter(x=>x==='L').length!==1) throw Error('layout 必须各含一个 E、m、L');
        const lastDecoder = slots.findLastIndex(x=>x.includes('t'));
        if (slots.length % pp) throw Error('layout 的 stage 数必须是 PP 的整数倍');
        return Array.from({length: pp}, (_, rank) => slots.map((label, i) => {
            let dense = 0, moe = 0;
            // Layer numbering follows virtual stage order, not physical rank order.
            const before = slots.slice(0, i).join('').split('t').length - 1;
            for (let j = 0; j < (label.match(/t/g) || []).length; j++) {
                if (before + j < denseLayers) dense++; else moe++;
            }
            return {label, dense, moe, mtp: (label.match(/m/g) || []).length,
                embedding: label.includes('E'), output: label.includes('L'), finalNorm: i===lastDecoder, index: i};
        }).filter((_, i) => i % pp === rank));
    }

    function validate(c, chunks) {
        const integerKeys = ['world','edpMultiplier','pp','tp','cp','ep','inner','group','seq','mbs','gbs','layers','hidden','ffn',
            'experts','topk','moeFfn','heads','headDim','qLora','oGroups','oLora','indexHeads','indexDim','indexTopk','ratio','window','vocab'];
        for (const k of integerKeys) if (!Number.isSafeInteger(c[k]) || c[k] < 1) throw Error(`${k} 必须是正整数`);
        for (const k of ['denseLayers','sharedFfn']) if (!Number.isSafeInteger(c[k]) || c[k] < 0) throw Error(`${k} 必须是非负整数`);
        for (const k of ['reserveGiB','nonTorchGiB','expertLoad']) if (!Number.isFinite(c[k]) || c[k] < 0) throw Error(`${k} 必须是非负数`);
        if (c.expertLoad < 1) throw Error('expertLoad 至少为 1（均衡接收量）');
        if (c.tp !== 1) throw Error('当前 DSv4 CSA 源码要求 TP=1；ETP 固定为 1');
        if (c.cp < 2) throw Error('本计算器仅建模经过 snapshot 验证的 packed THD CP>1 路径');
        if (c.mbs !== 1) throw Error('当前 packed THD CSA 路径要求 micro-batch-size=1');
        if (c.headDim !== 512) throw Error('当前 FlashMLA sparse 路径要求 headDim=512');
        if (![64,128].includes(c.heads) || ![512,1024].includes(c.indexTopk)) throw Error('仅支持本任务已核对的 heads=64/128、indexTopk=512/1024');
        if (c.ratio !== 4) throw Error('本任务只建模 compress ratio=4 的重叠 compressor');
        if (c.denseLayers > c.layers || c.topk > c.experts || c.heads % c.oGroups) throw Error('denseLayers/topk/oGroups 与模型维度不兼容');
        if (c.world % (c.tp*c.pp*c.cp) || c.world % (c.pp*c.ep) || c.experts % c.ep || c.seq % c.cp) throw Error('world、PP、CP、EP、experts 或 seq 无法整除');
        if (c.seq % (32*c.cp)) throw Error('本任务按每个 CP rank 的 token 容量为 32 的倍数建模');
        const domain = c.world / c.pp;
        if (domain % c.inner) throw Error('dense DP×CP 必须能被 FSDP inner 整除');
        if (typeof c.fp8Gather!=='boolean' || typeof c.paddedShards!=='boolean') throw Error('fp8Gather / paddedShards 必须是布尔值');
        if (typeof c.epOverlap!=='boolean') throw Error('epOverlap 必须是布尔值');
        if (!['optim','no_shard'].includes(c.denseOuter)) throw Error('denseOuter 仅支持 optim / no_shard');
        if (![1,2,3].includes(c.denseZeroStage)) throw Error('denseZeroStage 必须为 1、2 或 3');
        if (!Array.isArray(c.recompute) || c.recompute.some(x => !supportedRecompute.includes(x))) throw Error('不支持的 recompute 模块');
        if (c.recompute.includes('moe_fc1') && !c.recompute.includes('moe_act')) throw Error('moe_fc1 需要同时开启 moe_act');
        if (c.recompute.includes('csa_core_output') && !c.recompute.includes('mla_up_proj')) throw Error('csa_core_output 需要同时开启 mla_up_proj');
        const flat = chunks.flat();
        if (flat.reduce((n,x)=>n+x.dense+x.moe,0) !== c.layers || flat.reduce((n,x)=>n+x.mtp,0) !== 1 || flat.filter(x=>x.embedding).length !== 1 || flat.filter(x=>x.output).length !== 1) throw Error('layout 必须含指定层数、一个 E、一个 m 和一个 L');
        if (!chunks[0][0].embedding || !chunks.at(-1).at(-1).output || chunks.at(-1).at(-1).mtp !== 1) throw Error('本任务要求 E 在首 chunk，mL 同在末 chunk');
        const m = c.gbs / (domain/c.cp*c.mbs), v = chunks[0].length;
        if (m*v > 1000000) throw Error('microbatch × VPP 超过本简版计算器的回放上限 1,000,000');
        if (!Number.isInteger(m)) throw Error('GBS 必须能被 DP × MBS 整除');
        if (v > 1 && (c.group < c.pp || c.group > m || (m % c.group > 0 && m % c.group < c.pp))) throw Error(`当前调度无效：M=${m}, G=${c.group}, PP=${c.pp}；要求 PP≤G≤M，且 M%G=0 或 ≥PP`);
        return {dp: domain/c.cp, edp: domain/c.ep, vpp: v, microbatches: m, tokens: c.seq/c.cp, domain};
    }

    function savedLayer(c, moe) {
        const n = c.seq/c.cp, h = c.hidden, d = c.headDim, a = c.heads;
        const ih = c.indexHeads, id = c.indexDim, q = c.qLora;
        const rc = new Set(c.recompute), compact = n + 8*c.ratio;
        const compressed = compact/c.ratio, globalComp = compressed*c.cp;
        const rows = Math.ceil(n*c.topk*c.expertLoad/(c.experts/c.ep)/32)*32*(c.experts/c.ep);
        const x8 = 33/32; // one MXFP8 orientation plus per-32 E8M0 scales
        const items = {
            '残差 / 两次 norm 输入': 4*n*h,
            'CSA FlashMLA O + LSE（O 与 grouped projection 输入共享）': rc.has('csa_core_output') ? 0 : 2*n*a*d + 4*n*a,
            'CSA grouped O projection 输出': rc.has('csa_o_proj') ? 0 : 2*n*c.oGroups*c.oLora,
            'CSA compact hidden + boundary hidden': 2*compact*h + 2*c.window*h,
            'CSA Q LoRA、norm 与 index-Q MXFP8 输入': (4+x8)*n*q + 8*n,
            'CSA compressor FP32 pooling / norm 保存': 16*compact*(d+id) + 2*compressed*(d+id),
            'CSA gathered compressed KV': 2*globalComp*d,
            'CSA indexer 预计算 dQ/dK/dW': 2*n*ih*id + 2*globalComp*id + 2*n*ih,
            'CSA top-k int32 / lengths / position metadata': 4*n*Math.ceil((c.indexTopk+c.window)/128)*128 + 24*n,
        };
        if (rc.has('csa_core_output')) items['CSA O 重算：原始 top-k lengths'] = 4*n;
        if (!rc.has('mla_up_proj')) items['关闭 mla_up_proj：Q、KV、norm 输入及投影输入'] = 4*n*a*d + 4*n*d + x8*n*q;
        if (!rc.has('layernorm')) items['关闭 layernorm：norm 输出 / GEMM 输入'] = 4*n*h + 8*n;
        if (moe) {
            if (rc.has('moe_fc1')) {
                // FC1 replay retains BF16 dispatched input instead of only its MXFP8 column.
                items['MoE routed FC1 BF16 replay 输入'] = 2*rows*h;
            } else {
                items['MoE routed FC1 输出（moe_act 仍保存）'] = 4*rows*c.moeFfn;
                items['MoE routed FC1 MXFP8 输入（column + scales）'] = x8*rows*h;
            }
            const received = n*c.ep*(1-(1-1/c.ep)**c.topk)*c.expertLoad;
            items['MoE router / permutation / probs'] = 12*n*c.experts + 8*received*(c.experts/c.ep) + 8*rows;
            if (!rc.has('moe_act')) items['关闭 moe_act：GLU 中间量 / FC2 MXFP8 输入'] = (8+x8)*rows*c.moeFfn;
            // pre_dispatch node.detach stores the shared-expert side output until
            // backward, even when checkpointing shared_experts. It is separate
            // from the two residual/norm inputs already counted above.
            if (c.epOverlap && c.sharedFfn) items['combined 1F1B 共享专家侧输出'] = 2*n*h;
            if (c.sharedFfn && !rc.has('shared_experts')) items['关闭 shared_experts：共享 MLP 保存'] = x8*n*h + (10+x8)*n*c.sharedFfn;
        } else if (!rc.has('dense_fc1_act')) {
            items['Dense MLP MXFP8 输入 / quick_geglu 中间量'] = x8*n*h + (10+x8)*n*c.ffn;
        }
        return {items, total: sum(items), routedRows: rows};
    }

    function parameters(c, chunk) {
        const h=c.hidden, ah=c.heads*c.headDim, q=c.qLora;
        const count=chunk.dense+chunk.moe+chunk.mtp, moe=chunk.moe+chunk.mtp;
        const p8 = h*q + q*ah + h*c.headDim + c.oGroups*c.oLora*h + q*c.indexHeads*c.indexDim;
        const plain = ah*c.oLora + 4*h*(c.headDim+c.indexDim) + h*c.indexHeads;
        const norm = 2*h + q + 2*c.headDim + c.indexDim + c.heads;
        const ape = 8*(c.headDim+c.indexDim);
        const dense8 = count*p8 + chunk.dense*3*h*c.ffn + moe*3*h*c.sharedFfn + chunk.mtp*2*h*h;
        const densePlain = count*plain + moe*h*c.experts;
        const vocab = (Number(chunk.embedding) + Number(chunk.output) + Number(chunk.mtp>0))*c.vocab*h;
        const small = count*(norm+ape) + (chunk.finalNorm ? h : 0) + chunk.mtp*3*h;
        return {dense8, densePlain, vocab, small, fp32: count*(ape+c.heads),
            expert: moe*3*h*c.moeFfn*c.experts/c.ep};
    }

    function staticMemory(c, chunks, d) {
        const p={dense8:0,densePlain:0,vocab:0,small:0,fp32:0,expert:0};
        for (const chunk of chunks) add(p, parameters(c,chunk));
        const dense=p.dense8+p.densePlain+p.vocab+p.small;
        const optShard=c.denseOuter==='optim' ? d.domain : c.inner;
        const paramShard=c.denseZeroStage===3 ? c.inner : 1;
        const gradShard=c.denseZeroStage===1 ? 1 : c.inner;
        const p8=p.dense8+p.expert;
        const computeWidth=c.fp8Gather ? 2*33/32 : 2;
        const denseCompute=computeWidth*p.dense8+2*(p.densePlain+p.vocab+p.small)+2*p.fp32;
        const items={
            'compute weights（MXFP8 row+column+scales 或 BF16）': denseCompute/paramShard+computeWidth*p.expert,
            [c.denseZeroStage===1 ? 'dense FP32 main_grad（ZeRO-1，完整）' : 'dense FP32 main_grad / inner']: 4*dense/gradShard,
            'expert FP32 main_grad（ZeRO-1，完整）': 4*p.expert,
            'dense FP32 master + Muon / Adam states': (8*(p.dense8+p.densePlain)+12*(p.vocab+p.small))/optShard,
            'expert FP32 master + Muon momentum / EDP': 8*p.expert/d.edp,
        };
        if (!c.fp8Gather) items['BF16 gather 的 TE MXFP8 weight cache'] = 2*33/32*p8;
        return {items,total:sum(items),params:p};
    }

    // ZeRO-3 keeps its resident shard while gathering separate complete buffers.
    // Bound the active/prefetched roots and layer units separately: roots enclose
    // their child layers, and VPP transitions can prefetch the next chunk's root.
    // This is an envelope for the default one-successor prefetch, not trace replay.
    function parameterGatherWorkspace(c, groups) {
        if (c.denseZeroStage!==3 || c.inner===1) return 0;
        const units=new Map();
        for (const group of groups.filter(x=>x.kind==='dense')) {
            units.set(group.name,(units.get(group.name)||0)+group.fullComputeBytes);
        }
        const largestTwo=entries=>entries.map(([,bytes])=>bytes).sort((a,b)=>b-a).slice(0,2).reduce((a,b)=>a+b,0);
        return largestTwo([...units].filter(([name])=>name.endsWith('.root')))
            +largestTwo([...units].filter(([name])=>!name.endsWith('.root')));
    }

    // Replay ordinary (no MoE F/B overlap) interleaved scheduling, sampling after F too.
    function replay(pp, rank, v, m, group, weights) {
        const order=[];
        for(let start=0;start<m;start+=group) for(let c=0;c<v;c++) for(let j=start;j<Math.min(m,start+group);j++) order.push(c);
        const warmup=Math.min(order.length,v===1?pp-rank-1:2*(pp-rank-1)+(v-1)*group);
        const counts=Array(v).fill(0), states=[];
        const sample=(phase, chunk)=>states.push({counts:[...counts],saved:counts.reduce((s,n,i)=>s+n*weights[i],0),phase,chunk});
        for(let i=0;i<warmup;i++){counts[order[i]]++;sample('F',order[i]);}
        for(let i=0;i<order.length-warmup;i++){
            counts[order[warmup+i]]++;sample('F',order[warmup+i]);
            const b=v-1-order[i];counts[b]--;sample('B',b);
        }
        for(let i=order.length-warmup;i<order.length;i++){const b=v-1-order[i];counts[b]--;sample('B',b);}
        if(counts.some(x=>x!==0)||states.some(s=>s.counts.some(x=>x<0))) throw Error('流水线回放不闭合');
        return states;
    }

    // Default to four complete microbatch groups. A supplied G also determines
    // the default GBS; explicit GBS/G are preserved and validated by calculate.
    function batchSchedule(c) {
        const dp=c.world/(c.tp*c.pp*c.cp), group=c.group ?? c.pp;
        return {group, gbs:dp*c.mbs*4*group};
    }

    function calculate(config=defaults) {
        for (const key of Object.keys(config)) if (!(key in defaults) && key!=='edpMultiplier') throw Error(`未知配置字段：${key}`);
        const c={...defaults,...config};
        // Preserve legacy World-only JSON by inferring its EDP multiplier.
        // With an explicit multiplier, World is a derived output.
        if (!Object.hasOwn(config,'edpMultiplier') && Object.hasOwn(config,'world')) {
            if (!Number.isSafeInteger(config.world) || config.world<1) throw Error('world 必须是正整数');
            c.edpMultiplier=config.world/(c.pp*c.ep);
        }
        c.world=c.edpMultiplier*c.pp*c.ep;
        c.group=Object.hasOwn(config,'group') ? config.group : c.pp;
        const batchOverride=['world','edpMultiplier','pp','tp','cp','ep','mbs','group'].some(k=>Object.hasOwn(config,k));
        c.gbs=Object.hasOwn(config,'gbs') ? config.gbs : batchOverride ? batchSchedule(c).gbs : defaults.gbs;
        // Guard sizes before allocating layout arrays.
        if (!Number.isInteger(c.pp)||c.pp<1||c.pp>256) throw Error('PP 必须在 1..256');
        const chunks=parseLayout(c.layout,c.pp,c.denseLayers);
        // Layout is authoritative; accept legacy layers fields without using them.
        c.layers=chunks.flat().reduce((n,ch)=>n+ch.dense+ch.moe,0);
        const d=validate(c,chunks);
        const layers={dense:savedLayer(c,false),moe:savedLayer(c,true)};
        const warnings=[
            '本模型固定 BF16 + MXFP8 1D、TP1/ETP1、packed THD CP>1、CSA ratio4、fused RoPE、native CE、1 层 MTP；不外推到其他训练路径。',
            'dsa_selected_scores 在当前 THD CP fused CSA 分支不减少保存量：indexer 梯度在 forward 中计算并保存。',
            'expertLoad=1 为均衡路由；实际接收 token 数、FP8 padding、临时 workspace 与 allocator 碎片会改变峰值。',
        ];
        warnings.unshift(`当前批次：G=${c.group}，GBS=${c.gbs}，M=GBS/(DP×MBS)=${d.microbatches}。训练需使用相同批次配置；计算器不修改启动脚本。`);
        if(c.world===8&&c.denseOuter==='no_shard') warnings.push('EP2 proxy 使用 HSDP（dense outer=no_shard）；HFSDP 需要显式选择 outer=optim。');
        if(c.denseZeroStage!==2) warnings.push(`Dense ZeRO-${c.denseZeroStage} 为源码公式外推；现有 snapshot 校准覆盖 ZeRO-2。级别作用于 Inner，含 attention/shared/vocab，routed experts 仍为 ZeRO-1。`);
        if(c.denseZeroStage===3) warnings.push('ZeRO-3 另计最多两个 root 与两个 layer unit 的完整参数 gather buffers（默认预取一个后继）；MXFP8 按双方向计，覆盖重算需要。未模拟逐层通信时序或更大的预取窗口。');
        const stages=chunks.map((cs,rank)=>{
            const mem=staticMemory(c,cs,d),n=d.tokens,h=c.hidden;
            const layout=buffers.estimate(c,cs,d);
            const logicalStaticBytes=mem.total;
            if (layout.paddingBytes) mem.items['FSDP 布局 padding（compute / master / grad）']=layout.paddingBytes;
            mem.total+=layout.paddingBytes;
            const perChunk=cs.map(ch=>{
                const items={};add(items,layers.dense.items,ch.dense);add(items,layers.moe.items,ch.moe+ch.mtp);
                items['chunk 输入 / pipeline buffer']=2*n*h;
                if(ch.mtp) items['MTP embedding / concat / norms / projection 保存']=12*n*h;
                if(ch.output) items['主 loss + MTP loss：FP32 softmax']=8*n*c.vocab;
                return items;
            });
            const states=replay(c.pp,rank,d.vpp,d.microbatches,c.group,perChunk.map(sum));
            const peak=states.reduce((a,b)=>b.saved>a.saved?b:a);
            const savedItems={};peak.counts.forEach((k,i)=>add(savedItems,perChunk[i],k));
            // Separate an estimated forward/attention workspace from an RS upper envelope.
            // Recomputed Q/projection/norm + dO + undo-RoPE O/dO + cuDNN dQ.
            // Six BF16 [N,heads,D] planes are visible at rank-0 snapshot peak.
            // O/LSE are restored for one active backward layer, no longer saved
            // across every outstanding layer/microbatch when output replay is on.
            const outputReplayBytes=c.recompute.includes('csa_core_output') ? 2*n*c.heads*c.headDim+4*n*c.heads : 0;
            const attnWorkspace=12*n*c.heads*c.headDim + 2*n*c.indexHeads*c.indexDim + outputReplayBytes;
            const lossWorkspace=cs.some(x=>x.output)?2*n*c.vocab:0;
            const computeWorkspaceBytes=Math.max(attnWorkspace,lossWorkspace);
            const parameterGatherBytes=parameterGatherWorkspace(c,layout.details);
            const gradients=buffers.gradientWorkspace(c,layout.details);
            const fsdpTemporaryItems={
                '完整 partial_grad staging（两个 unit，包含 root / experts）': gradients.partialBytes,
                '当前 unit 的完整 autograd wgrad': gradients.autogradBytes,
                'ZeRO-3 参数 gather / prefetch': parameterGatherBytes,
            };
            const fsdpTemporaryBytes=sum(fsdpTemporaryItems);
            // Staging can survive through the next layer's attention computation.
            // Count it alongside compute workspaces, not only in a separate reserve.
            const transientBytes=computeWorkspaceBytes+fsdpTemporaryBytes;
            const totalBytes=mem.total+peak.saved+transientBytes;
            return {rank,chunks:cs,peakCounts:peak.counts,staticItems:mem.items,savedItems,
                staticBytes:mem.total,logicalStaticBytes,paddingBytes:layout.paddingBytes,bufferGroups:layout.details,
                savedBytes:peak.saved,computeWorkspaceBytes,parameterGatherBytes,transientBytes,totalBytes,
                fsdpTemporaryItems,fsdpTemporaryBytes,gradientWorkspace:gradients,
                rsWorkspaceBytes:gradients.bytes,
                budgetBytes:totalBytes+(c.reserveGiB+c.nonTorchGiB)*GiB};
        });
        if (c.denseZeroStage===1) warnings.push('ZeRO-1 按无 FSDP padded-shard 增量的场景计算，padding 开关不增加显存；复现此场景需关闭训练侧 padded shards。');
        else if (c.paddedShards) warnings.push('已按每个 FSDP unit / dtype 的实际矩阵行宽计算 padded buffer；对齐可能增加常驻 padding。allocated 另包含两个完整 unit 的 staging，可能不与激活峰值同时发生。');
        else warnings.push('关闭 padded shards 是独立配置：启动需补 --no-megatron-fsdp-padded-shards；本计算使用平均 optimizer shard，其他规模需要独立验证。');
        warnings.push('FSDP 临时量按两个完整 unit 的 FP32 staging + 一个 unit 的 autograd 梯度估算，EDP=1 的专家仍会打包。root、wgrad、保存激活的各自最大值未必同时发生，尤其末 stage 可能偏保守；不包含完整 Muon NS 临时量。');
        return {config:c,derived:d,warnings,layers,stages,maxStage:stages.reduce((a,b)=>b.budgetBytes>a.budgetBytes?b:a)};
    }

    function compare(config=defaults) {
        const current=calculate(config), c=current.config;
        const zeroComparison=[1,2,3].map(denseZeroStage=>{
            const run=denseZeroStage===c.denseZeroStage ? current : calculate({...c,denseZeroStage});
            const stages=run.stages.map(({rank,totalBytes,budgetBytes})=>({rank,totalBytes,budgetBytes}));
            return {denseZeroStage,stages,maxAllocated:stages.reduce((a,b)=>b.totalBytes>a.totalBytes?b:a)};
        });
        const other=calculate({...c,paddedShards:!c.paddedShards});
        const [on,off]=c.paddedShards ? [current,other] : [other,current];
        // Compare the SAME rank with/without padding; the global peak rank can move.
        const stages=on.stages.map((s,i)=>({rank:s.rank,
            totalBytes:s.totalBytes-off.stages[i].totalBytes,
            staticBytes:s.staticBytes-off.stages[i].staticBytes,
            parameterGatherBytes:s.parameterGatherBytes-off.stages[i].parameterGatherBytes,
            gradientWorkspaceBytes:s.rsWorkspaceBytes-off.stages[i].rsWorkspaceBytes,
            budgetBytes:s.budgetBytes-off.stages[i].budgetBytes}));
        const paddingImpact={stages,
            maxAllocated:stages.reduce((a,b)=>b.totalBytes>a.totalBytes?b:a),
            maxBudget:stages.reduce((a,b)=>b.budgetBytes>a.budgetBytes?b:a)};
        return {...current,zeroComparison,paddingImpact};
    }
    const api={presets:{hfsdp16,local},recomputeOrigins,calculate,compare,batchSchedule,parseLayout,replay,savedLayer,parameters,GiB};
    if(typeof module!=='undefined'&&module.exports) module.exports=api;
    else root.CSAMemory=api;
    if(typeof require!=='undefined'&&require.main===module){
        const fs=require('node:fs');
        const cfg=process.argv[2] ? (api.presets[process.argv[2]]||JSON.parse(fs.readFileSync(process.argv[2],'utf8'))) : defaults;
        process.stdout.write(JSON.stringify(calculate(cfg),null,2)+'\n');
    }
})(typeof globalThis==='undefined'?this:globalThis);
