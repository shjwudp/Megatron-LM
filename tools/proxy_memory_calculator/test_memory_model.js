// SPDX-License-Identifier: Apache-2.0
// Run with: node --test tools/proxy_memory_calculator/test_memory_model.js
const test = require('node:test');
const assert = require('node:assert/strict');
const {calculate, compare, presets, GiB, replay} = require('./memory_model.js');
const withoutO = presets.hfsdp16.recompute.filter(name => name !== 'csa_core_output');

test('16-node estimates include FSDP buffers and agree with measured PP maxima within 5%', () => {
    const evidence = require('./fsdp16_memory_validation.json');
    const result = calculate(evidence.config);
    assert.equal(result.derived.edp, 1);
    assert.equal(result.derived.microbatches, 8);
    assert.deepEqual(calculate(presets.hfsdp16).stages, result.stages);
    for (const measured of evidence.stages) {
        const stage = result.stages[measured.pp];
        assert.ok(Math.abs(stage.totalBytes / GiB / measured.max_allocated_GiB - 1) < 0.05);
        // The observed live staging is from snapshot18, not max-per-trace sums.
        assert.ok(stage.gradientWorkspace.partialBytes / GiB >= measured.snapshot.fsdp_partial_at_peak_GiB);
        assert.ok(stage.gradientWorkspace.partialBytes / GiB - measured.snapshot.fsdp_partial_at_peak_GiB < 0.4);
        assert.equal(stage.budgetBytes-stage.totalBytes, 28*GiB);
    }
});

test('FSDP staging includes root dtype buckets and expert buffers even with EDP=1', () => {
    const c = presets.hfsdp16, result = calculate(c);
    const root = result.stages[0].gradientWorkspace.units.find(x => x.name === 'chunk0.root');
    assert.equal(root.partialBytes, 4*c.vocab*c.hidden);
    assert.equal(root.autogradBytes, 2*c.vocab*c.hidden);
    const expert = result.stages[1].gradientWorkspace.units.find(x => x.kind === 'expert');
    assert.equal(expert.partialBytes, 4*3*c.hidden*c.moeFfn*(c.experts/c.ep));
    const shorter = calculate({...c, seq:16384, inner:4});
    assert.deepEqual(shorter.stages.map(s=>s.fsdpTemporaryBytes), result.stages.map(s=>s.fsdpTemporaryBytes));
    const noOverlap = calculate({...c, epOverlap:false});
    assert.equal(result.layers.moe.total-noOverlap.layers.moe.total, 2*(c.seq/c.cp)*c.hidden);
    assert.equal(calculate({...c,edpMultiplier:2}).config.world, 2*c.world);
});

test('O replay replaces outstanding O/LSE with per-layer backward workspace', () => {
    const base={...presets.local,denseOuter:'optim',recompute:withoutO};
    const before=calculate(base), after=calculate({...base,recompute:presets.hfsdp16.recompute});
    const n=base.seq/base.cp, bytes=2*n*base.heads*base.headDim+4*n*base.heads;
    assert.equal(before.layers.moe.total-after.layers.moe.total,bytes-4*n);
    assert.equal(after.layers.moe.items['CSA FlashMLA O + LSE（O 与 grouped projection 输入共享）'],0);
    for (let i=0;i<before.stages.length;i++) {
        assert.deepEqual(after.stages[i].staticItems,before.stages[i].staticItems);
        assert.equal(after.stages[i].computeWorkspaceBytes-before.stages[i].computeWorkspaceBytes,bytes);
    }
    assert.throws(()=>calculate({...base,recompute:['csa_core_output']}),/mla_up_proj/);
    const {recomputeOrigins}=require('./memory_model.js');
    assert.equal(Object.values(recomputeOrigins).filter(x=>x==='main 原有').length,5);
    assert.equal(Object.values(recomputeOrigins).filter(x=>x==='本分支新增').length,4);
});

test('legacy O replay base estimates retain their historical 5% calibration', () => {
    const evidence=require('./csa_output_validation.json');
    assert.equal(evidence.snapshots.length,16);
    for (const sample of evidence.snapshots) {
        const recompute=sample.case==='baseline'?withoutO:presets.hfsdp16.recompute;
        const predicted=calculate({...presets.local,denseOuter:'optim',paddedShards:true,recompute}).stages[sample.pp_rank];
        assert.ok(Math.abs((predicted.totalBytes-predicted.rsWorkspaceBytes)/sample.peak_fc_bytes-1)<0.05,
            `${sample.case} rank=${sample.rank}: ${predicted.totalBytes/GiB} vs ${sample.peak_fc_bytes/GiB}`);
    }
});

test('reject actual runtime-invalid and out-of-scope configurations', () => {
    for (const changes of [{edpMultiplier:0}, {tp:2}, {cp:1}, {mbs:2}, {headDim:256},
        {layout:'EEtt|tt|tt|ttmL'}, {recompute:['core_attn']}, {expertLoad:0.9},
        {paddedShards:'false'}, {cp:0}, {pp:5}, {seq:32760}, {misspelledHidden:2048}]) {
        assert.throws(()=>calculate({...presets.hfsdp16,...changes}), JSON.stringify(changes));
    }
});

test('CSA aliases one attention O storage and keeps indexer forward gradients', () => {
    const r=calculate(presets.local), terms=r.layers.moe.items;
    assert.equal(terms['CSA FlashMLA O + LSE（O 与 grouped projection 输入共享）'],GiB+4*2**20);
    assert.equal(terms['CSA grouped O projection 输出'],128*2**20);
    const noScores=calculate({...presets.local,recompute:presets.local.recompute.filter(x=>x!=='dsa_selected_scores')});
    assert.equal(noScores.layers.moe.total,r.layers.moe.total);
    for(const name of ['mla_up_proj','layernorm','moe_act','shared_experts']) {
        const off=calculate({...presets.local,recompute:presets.local.recompute.filter(x=>x!==name)});
        assert.ok(off.layers.moe.total>r.layers.moe.total,name);
    }
});

test('EP shards parameters while balanced routed activation volume stays similar', () => {
    const a=calculate(presets.local), b=calculate({...presets.local,ep:4});
    assert.ok(b.stages[1].staticBytes<a.stages[1].staticBytes);
    const term='MoE routed FC1 输出（moe_act 仍保存）';
    assert.equal(a.layers.moe.items[term],b.layers.moe.items[term]);
    assert.ok(calculate({...presets.local,expertLoad:1.25}).layers.moe.total>a.layers.moe.total);
});

test('no-VPP and all-warmup replay close and respect finite microbatch counts', () => {
    const normal=replay(4,0,1,8,4,[1]);
    assert.equal(Math.max(...normal.map(s=>s.saved)),4);
    const short=replay(4,0,1,2,4,[1]);
    assert.equal(Math.max(...short.map(s=>s.saved)),2);
    assert.deepEqual(short.at(-1).counts,[0]);
});

test('legacy local base estimates retain their historical 5% calibration', () => {
    const measured={true:[43.1327654123,66.7721147463],false:[44.5681430437,69.0121365674]};
    for(const fp8Gather of [true,false]){
        const r=calculate({...presets.local,fp8Gather,paddedShards:true});
        r.stages.forEach((s,i)=>{
            const actual=measured[fp8Gather][i], predicted=(s.totalBytes-s.rsWorkspaceBytes)/GiB;
            assert.ok(Math.abs(predicted/actual-1)<0.05,`FP8=${fp8Gather} rank=${i}: ${predicted} vs ${actual}`);
        });
    }
});

test('padding is opt-in and honors the actual MFSDP BlockAtomic padding quantum', () => {
    const a=calculate({edpMultiplier:4,paddedShards:true}), b=calculate({edpMultiplier:4});
    const group=a.stages[1].bufferGroups.find(x=>x.name.endsWith('.moe0')&&x.dtype==='mxfp8');
    // Dense units must split evenly into complete row-atomic blocks on every shard.
    assert.equal(group.allocatedElements % (group.alignmentElements*a.derived.domain),0);
    assert.ok(group.allocatedElements>=group.logicalElements);
    assert.ok(a.stages[1].staticBytes>b.stages[1].staticBytes);
    assert.equal(a.stages[0].savedBytes,b.stages[0].savedBytes);
    assert.equal(b.stages[0].paddingBytes,0);
    for (const preset of Object.values(presets)) {
        assert.equal(preset.paddedShards,false);
        assert.ok(calculate(preset).stages.every(s=>s.paddingBytes===0));
    }
});

test('legacy recompute base estimates retain their historical 5% calibration', () => {
    const evidence=require('./recompute_validation.json');
    assert.equal(evidence.snapshots.length,16);
    for (const sample of evidence.snapshots) {
        const recompute=sample.case==='baseline-v3' ? presets.local.recompute : withoutO;
        const predicted=calculate({...presets.local,denseOuter:'optim',paddedShards:true,recompute}).stages[sample.pp_rank];
        assert.ok(Math.abs((predicted.totalBytes-predicted.rsWorkspaceBytes)/sample.peak_fc_bytes-1)<0.05,
            `${sample.case} rank=${sample.rank}: ${predicted.totalBytes/GiB} vs ${sample.peak_fc_bytes/GiB}`);
    }
    assert.throws(()=>calculate({...presets.hfsdp16,recompute:['moe_fc1']}),/moe_act/);
});

test('Dense ZeRO levels leave expert state and saved activations unchanged', () => {
    for (const fp8Gather of [true,false]) for (const denseOuter of ['optim','no_shard']) {
        const runs=[1,2,3].map(denseZeroStage=>calculate({...presets.local,fp8Gather,denseOuter,denseZeroStage}));
        for (let i=0;i<runs[0].stages.length;i++) {
            const [z1,z2,z3]=runs.map(r=>r.stages[i]);
            assert.ok(z1.staticBytes>z2.staticBytes && z2.staticBytes>z3.staticBytes);
            for (const stage of [z1,z3]) {
                assert.deepEqual(stage.savedItems,z2.savedItems);
                assert.deepEqual(stage.peakCounts,z2.peakCounts);
                assert.deepEqual(stage.bufferGroups.filter(g=>g.kind==='expert'),z2.bufferGroups.filter(g=>g.kind==='expert'));
                assert.equal(stage.staticItems['dense FP32 master + Muon / Adam states'],z2.staticItems['dense FP32 master + Muon / Adam states']);
                assert.equal(stage.staticItems['BF16 gather 的 TE MXFP8 weight cache'],z2.staticItems['BF16 gather 的 TE MXFP8 weight cache']);
            }
            assert.equal(z1.parameterGatherBytes,0);
            assert.equal(z2.parameterGatherBytes,0);
            assert.ok(z3.parameterGatherBytes>0);
            assert.equal(z3.transientBytes-z3.computeWorkspaceBytes,z3.fsdpTemporaryBytes);
            assert.equal(z3.fsdpTemporaryBytes,z3.parameterGatherBytes+z3.rsWorkspaceBytes);
        }
    }
    for (const denseZeroStage of [0,4,'3',null]) assert.throws(()=>calculate({denseZeroStage}),/denseZeroStage/);
    assert.deepEqual(calculate(),calculate({denseZeroStage:2}));
});

test('ZeRO-3 counts root and prefetched layer gathers once, including in the budget', () => {
    const c={...presets.local,denseZeroStage:3,paddedShards:false};
    const r=calculate(c), longerSequence=calculate({...c,seq:2*c.seq});
    r.stages.forEach((s,i)=>{
        const root=s.bufferGroups.filter(g=>g.name.endsWith('.root')).reduce((n,g)=>n+g.fullComputeBytes,0);
        const largestLayer=Math.max(...s.bufferGroups.filter(g=>g.kind==='dense'&&!g.name.endsWith('.root')).map(g=>g.fullComputeBytes));
        assert.ok(s.parameterGatherBytes>=root+largestLayer);
        assert.equal(s.parameterGatherBytes,longerSequence.stages[i].parameterGatherBytes);
        assert.ok(s.budgetBytes>=s.staticBytes+s.savedBytes+s.parameterGatherBytes+s.rsWorkspaceBytes);
    });
    // Inner=1 removes collectives, but MFSDP still packs full gradients.
    const unsharded=[1,2,3].map(denseZeroStage=>calculate({...c,denseZeroStage,inner:1}));
    assert.deepEqual(unsharded[0].stages.map(s=>s.staticBytes),unsharded[1].stages.map(s=>s.staticBytes));
    assert.ok(unsharded[1].stages.every(s=>s.rsWorkspaceBytes>0));
    assert.deepEqual(unsharded[1].stages.map(s=>s.totalBytes),unsharded[2].stages.map(s=>s.totalBytes));
    assert.ok(unsharded[2].stages.every(s=>s.parameterGatherBytes===0));
});

test('ZeRO-1 baseline has no padding in resident buffers or temporary workspaces', () => {
    for (const preset of [presets.hfsdp16,presets.local]) for (const fp8Gather of [true,false]) {
        const c={...preset,fp8Gather,denseZeroStage:1};
        const off=calculate({...c,paddedShards:false}), on=calculate({...c,paddedShards:true});
        assert.deepEqual(on.stages,off.stages);
        for (const stage of on.stages) {
            assert.equal(stage.paddingBytes,0);
            assert.ok(stage.bufferGroups.every(b=>b.allocatedElements===b.logicalElements));
        }
    }
});

test('comparison includes each rank and measures padding deltas on the same rank', () => {
    for (const denseZeroStage of [1,2,3]) for (const paddedShards of [false,true]) {
        const config={...{...presets.hfsdp16,recompute:withoutO},denseZeroStage,paddedShards};
        const r=compare(config);
        for (const entry of r.zeroComparison) {
            const expected=calculate({...config,denseZeroStage:entry.denseZeroStage});
            assert.deepEqual(entry.stages.map(s=>s.totalBytes),expected.stages.map(s=>s.totalBytes));
            assert.equal(entry.maxAllocated.totalBytes,Math.max(...entry.stages.map(s=>s.totalBytes)));
        }
        const off=calculate({...config,paddedShards:false}), on=calculate({...config,paddedShards:true});
        for (const delta of r.paddingImpact.stages) {
            assert.equal(delta.totalBytes,on.stages[delta.rank].totalBytes-off.stages[delta.rank].totalBytes);
            assert.equal(delta.budgetBytes,on.stages[delta.rank].budgetBytes-off.stages[delta.rank].budgetBytes);
            assert.ok(Math.abs(delta.totalBytes-delta.staticBytes-delta.parameterGatherBytes-delta.gradientWorkspaceBytes)<0.001);
        }
        assert.equal(r.paddingImpact.maxAllocated.totalBytes,Math.max(...r.paddingImpact.stages.map(s=>s.totalBytes)));
        assert.equal(r.paddingImpact.maxBudget.budgetBytes,Math.max(...r.paddingImpact.stages.map(s=>s.budgetBytes)));
        if (denseZeroStage===1) assert.equal(r.paddingImpact.maxAllocated.totalBytes,0);
        if (denseZeroStage===3) assert.ok(r.paddingImpact.maxAllocated.parameterGatherBytes>0);
    }
    assert.throws(()=>compare({denseZeroStage:4}),/denseZeroStage/);
    assert.throws(()=>compare({unknown:true}),/未知配置字段/);
});


test('only independent 16-node and EP2 proxy presets are exposed', () => {
    assert.deepEqual(Object.keys(presets).sort(), ['hfsdp16', 'local']);
    assert.deepEqual(calculate().config, calculate(presets.hfsdp16).config);
    assert.equal(calculate(presets.hfsdp16).config.layers, 44);
    assert.equal(calculate(presets.local).config.layers, 8);
    assert.equal(calculate(presets.hfsdp16).derived.edp, 1);
    assert.equal(calculate(presets.local).derived.edp, 2);
    for (const preset of Object.values(presets)) {
        assert.equal(preset.world, preset.edpMultiplier*preset.pp*preset.ep);
        assert.equal(preset.layers, [...preset.layout.replace(/\(tttt\|\)\*10/, 'tttt|'.repeat(10))].filter(x=>x==='t').length);
    }
});

test('layout controls decoder depth and excludes MTP', () => {
    const r=calculate({...presets.local,layout:'E(tt|)*3ttmL',layers:999});
    assert.equal(r.config.layers,8);
    assert.equal(r.stages.flatMap(s=>s.chunks).reduce((n,ch)=>n+ch.mtp,0),1);
    assert.throws(()=>calculate({...r.config,denseLayers:9}),/denseLayers/);
    assert.deepEqual(calculate({...r.config,layers:999}),r);
});

test('editable topology derives World and default batch groups', () => {
    for (const extra of [{}, {edpMultiplier:4}, {ep:4}, {cp:4}]) {
        const c={...presets.local,...extra}; delete c.group; delete c.gbs;
        const r=calculate(c);
        assert.equal(r.config.world,r.config.edpMultiplier*r.config.pp*r.config.ep);
        assert.equal(r.config.group,r.config.pp);
        assert.equal(r.derived.microbatches,4*r.config.group);
    }
    const legacy={...presets.local,world:16}; delete legacy.edpMultiplier;
    assert.equal(calculate(legacy).derived.edp,4);
    for (const bad of [0,1.5,'2']) assert.throws(()=>calculate({...presets.local,edpMultiplier:bad}));
});

test('manual batch overrides survive topology changes and invalid tails fail', () => {
    const c={...presets.local,group:3,gbs:18};
    const r=calculate(c);
    assert.equal(r.derived.microbatches,9);
    assert.equal(r.config.group,3);
    assert.equal(compare(c).config.gbs,18);
    assert.equal(calculate({...c,cp:4}).config.gbs,18);
    assert.ok(calculate({...presets.local,gbs:4}).stages[0].savedBytes<calculate(presets.local).stages[0].savedBytes);
    for (const bad of [{group:3,gbs:14},{group:0},{gbs:0},{gbs:3},{group:3.5}])
        assert.throws(()=>calculate({...presets.local,...bad}));
});
