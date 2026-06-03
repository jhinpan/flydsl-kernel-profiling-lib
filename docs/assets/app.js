/* FlyDSL Kernel Atlas — dashboard logic (no deps, reads window.KERNEL_DATA) */
(function(){
const D = window.KERNEL_DATA;
if(!D){document.body.innerHTML='<p style="padding:40px;font-family:monospace">data/data.js failed to load.</p>';return;}
const K = D.kernels;
const $ = (s,r=document)=>r.querySelector(s);
const el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};

const STALL_COLORS={
 'VMEM-wait':'#4a8dff','VMEM-load':'#ffc24b','VMEM-store':'#ff9457','LDS/SMEM-wait':'#9b8cff',
 'MFMA/FMA':'#33e0c8','barrier':'#ff6275','LDS':'#6ad27f','SMEM':'#3a4a63','other':'#5b6b82'};
const sc=t=>STALL_COLORS[t]||'#5b6b82';

function verdictClass(v){return v==='FlyDSL faster'?'good':v==='comparable'?'warn':v==='FlyDSL slower'?'bad':'none';}
function verdictBadge(v){const c=verdictClass(v);const lbl=v==='FlyDSL faster'?'★ FlyDSL wins':v==='comparable'?'≈ comparable':v==='FlyDSL slower'?'△ headroom':'no baseline';
 return `<span class="verdict v-${c}">${lbl}</span>`;}
function accentColor(v){return v==='FlyDSL faster'?'var(--good)':v==='comparable'?'var(--warn)':v==='FlyDSL slower'?'var(--bad)':'var(--slate)';}
const f1=x=>x==null?'—':(Math.abs(x)>=100?Math.round(x):x.toFixed(1));
const fx=x=>x==null?'—':x.toFixed(2)+'×';

/* ---- multishape verdict helpers ---- */
const MS_VERDICT={
 promote:{cls:'promote',lbl:'▲ promote'},
 parity:{cls:'parity',lbl:'≈ parity'},
 tune_needed:{cls:'tune',lbl:'◆ tune'},
 rewrite_needed:{cls:'rewrite',lbl:'✕ rewrite'},
 baseline_blocked:{cls:'blocked',lbl:'⊘ base-blocked'},
 blocked:{cls:'blocked',lbl:'⊘ blocked'},
};
function msInfo(v){return MS_VERDICT[v]||{cls:'blocked',lbl:v||'—'};}
function msBadge(v){const i=msInfo(v);return `<span class="verdict v-${i.cls}">${i.lbl}</span>`;}
// geomean of best-correct-baseline / flydsl -> >1 means FlyDSL faster
const msGeo=g=>g==null?'—':g.toFixed(g>=10?0:2)+'×';

/* ---- header chips + kpis ---- */
function header(){
 const p=D.provenance;
 $('#chips').innerHTML=[
  `<span class="chip"><span class="dot"></span>${D.summary.with_att} ATT bundles · 8× MI350X</span>`,
  `<span class="chip">FlyDSL <b>${p.flydsl_version}</b> @ ${p.flydsl_commit}</span>`,
  `<span class="chip">ROCm <b>${p.rocm}</b> · rocprofv3 ${p.rocprofv3}</span>`,
  `<span class="chip">arch <b>gfx950</b> · CDNA4</span>`,
  `<span class="chip">captured <b>${p.capture_date}</b></span>`,
 ].join('');
 const s=D.summary;
 const ms=D.multishape_summary;
 $('#kpis').innerHTML=[
  ['fly',ms?ms.kernels:s.with_att,'kernels · multishape'],
  ['',ms?ms.total_shapes:s.with_att,'shapes benchmarked'],
  ['good',ms?(ms.promote.length+ms.parity.length):s.wins,'promote + parity'],
  ['bad',ms?(ms.tune_needed.length+ms.rewrite_needed.length):s.headroom,'tune / rewrite'],
  ['',s.with_att,'rocprof ATT traces'],
 ].map(([c,n,l])=>`<div class="kpi ${c}"><div class="n">${n}</div><div class="l">${l}</div></div>`).join('');
 // metric legend band
 if(ms){
  const models3='Qwen / DeepSeek / Kimi';
  $('#metricbar').innerHTML=
   `<span><b>${ms.kernels}</b> kernels · <b>${ms.total_shapes}</b> shapes · <b>3</b> models (${models3})</span>`+
   `<span class="leg">metric = <code>${escapeHTML(ms.metric)}</code></span>`+
   `<span class="leg">geomean &gt; <code>1.00×</code> ⇒ FlyDSL faster than best correct baseline</span>`;
 }
}

/* ---- speedup chart ---- */
function speedChart(){
 const rows=K.filter(k=>k.speedup_vs_baseline!=null).sort((a,b)=>b.speedup_vs_baseline-a.speedup_vs_baseline);
 $('#spd-tag').textContent=`${rows.length} kernels with a wired baseline · log-ish scale`;
 const wrap=$('#spdchart');wrap.innerHTML='';
 const MAXR=3; // visual clamp for ratio>3
 rows.forEach(k=>{
  const sp=k.speedup_vs_baseline;
  const v=verdictClass(k.verdict),col=accentColor(k.verdict);
  // map ratio to 0..100 with center=1.0 at 50%
  let pct;
  if(sp>=1){pct=50+Math.min((sp-1)/(MAXR-1),1)*48;}
  else{pct=50-Math.min((1-sp)/(1-1/MAXR),1)*48;}
  const left=Math.min(50,pct),width=Math.abs(pct-50);
  const row=el('div','bar-row');
  row.innerHTML=`<div class="bk">${k.name}<span class="cat">${k.op_category} · vs ${shorten(k.baseline_name)}</span></div>
   <div class="track"><div class="mid"></div><div class="fillbar" style="left:${left}%;width:${width}%;background:${col}"></div></div>
   <div class="bv" style="color:${col}">${fx(sp)}</div>`;
  row.onclick=()=>openDrawer(k.stem);
  wrap.appendChild(row);
 });
}
function shorten(s){if(!s)return 'baseline';s=s.replace(/\(.*$/,'').trim();return s.length>26?s.slice(0,24)+'…':s;}

/* ---- grid ---- */
let activeCat='all',q='',sortBy='cat',msFilter='all';
function catPills(){
 const cats=['all',...Object.keys(D.summary.categories).sort()];
 $('#catpills').innerHTML=cats.map(c=>`<span class="pill${c==='all'?' on':''}" data-c="${c}">${c}${c!=='all'?` ${D.summary.categories[c]}`:''}</span>`).join('');
 $('#catpills').querySelectorAll('.pill').forEach(p=>p.onclick=()=>{
  activeCat=p.dataset.c;$('#catpills').querySelectorAll('.pill').forEach(x=>x.classList.toggle('on',x===p));renderGrid();});
}
const msGeoOf=k=>{const m=k.multishape;if(!m||m.geomean_vs_best==null)return null;
 if(m.verdict==='blocked'||m.verdict==='baseline_blocked')return null; // artifact: keep out of the geomean ranking
 return m.geomean_vs_best;};
function sortFn(a,b){
 if(sortBy==='spd')return (b.speedup_vs_baseline??-1)-(a.speedup_vs_baseline??-1);
 if(sortBy==='msgeo')return (msGeoOf(b)??-1)-(msGeoOf(a)??-1);
 if(sortBy==='us')return (b.flydsl_us??-1)-(a.flydsl_us??-1);
 if(sortBy==='occ')return (parseInt(b.occupancy)||0)-(parseInt(a.occupancy)||0);
 if(sortBy==='stall')return (b.stall_pct_total??-1)-(a.stall_pct_total??-1);
 return (a.op_category||'z').localeCompare(b.op_category||'z')||a.name.localeCompare(b.name);
}
function renderGrid(){
 const g=$('#grid');g.innerHTML='';
 let list=K.filter(k=>(activeCat==='all'||k.op_category===activeCat)
   &&(msFilter==='all'||(k.multishape&&k.multishape.verdict===msFilter))
   &&(!q||k.name.toLowerCase().includes(q)||(k.jit_kernel||'').toLowerCase().includes(q)));
 list.sort(sortFn);
 $('#grid-tag').textContent=`${list.length} shown`;
 list.forEach(k=>{
  const col=accentColor(k.verdict);
  const sb=stallBarHTML(k.stall_breakdown,7);
  const top=(k.top_source_lines&&k.top_source_lines[0])?k.top_source_lines[0].source:'';
  const c=el('div','card');
  c.innerHTML=`<div class="accent" style="background:${col}"></div>
   <div class="body">
    <div class="top"><div><h3>${k.name}</h3><div class="catb">${k.op_category}${k.jit_kernel?' · '+truncMid(k.jit_kernel,22):''}</div></div>${verdictBadge(k.verdict)}</div>
    <div class="ml" style="color:var(--ink3);font-family:var(--mono);font-size:10px;letter-spacing:.04em;margin-bottom:2px">rocprof · 1 shape</div>
    <div class="metrics">
      <div class="metric"><div class="mv fly">${f1(k.flydsl_us)}<span style="font-size:12px;color:var(--ink3)"> µs</span></div><div class="ml">FlyDSL${k.baseline_us?` · base ${f1(k.baseline_us)}µs`:''}</div></div>
      <div class="metric"><div class="mv">${k.tflops?f1(k.tflops):(k.bandwidth_gbs?f1(k.bandwidth_gbs/1000):'—')}</div><div class="ml">${k.tflops?'TFLOPS':k.bandwidth_gbs?'TB/s':'—'}</div></div>
      <div class="metric"><div class="mv">${k.occupancy??'—'}</div><div class="ml">waves/SIMD</div></div>
    </div>
    ${k.has_bundle?`<div class="ml" style="color:var(--ink3);margin-bottom:3px">stall ${k.stall_pct_total?Math.round(k.stall_pct_total)+'%':'—'} · top: ${k.top_stall_type||'—'}</div>${sb}`:`<div class="badge-err">${k.status==='compile_fail'?'compile fail':(k.capture_error||'no ATT')}</div>`}
    ${msStripHTML(k)}
    <div class="cardfoot"><span class="src">${top?truncMid(top,30):''}</span><span>${k.ins?k.ins+' ins · '+k.mapped_pct+'%':''}</span></div>
   </div>`;
  c.onclick=()=>openDrawer(k.stem);
  g.appendChild(c);
 });
}
function stallBarHTML(sb,h){
 if(!sb||!sb.length)return '';
 const segs=sb.filter(s=>s.pct>0.4).map(s=>`<span style="width:${s.pct}%;background:${sc(s.type)}" title="${s.type} ${s.pct}%"></span>`).join('');
 return `<div class="stallbar" style="height:${h}px">${segs}</div>`;
}
function msModelsChip(ms){
 const mdl=ms.models||[];
 if(!mdl.length)return '';
 const title=escapeHTML(mdl.join(', '));
 return `<span class="mschip" title="${title}">${mdl.length} model${mdl.length===1?'':'s'}</span>`;
}
function msStripHTML(k){
 const ms=k.multishape;
 const i=ms?msInfo(ms.verdict):msInfo('blocked');
 if(!ms){
  // no multishape record at all (e.g. preshuffle_gemm_v2): render gracefully
  return `<div class="msrow" onclick="event.stopPropagation()">
    <div class="msv" style="color:var(--ink3)">—</div>
    <div class="msmeta"><span class="mslbl">multishape cold-cache</span>
      <span class="mssub">no multishape run</span></div>
    <span class="verdict v-blocked">⊘ n/a</span></div>`;
 }
 const geo=ms.geomean_vs_best;
 // blocked / baseline_blocked: the number is a slow-eager-baseline artifact, not a
 // real win -- render neutral (never green) so moe_gemm's 158x isn't a fake "win".
 const msBlk=ms.verdict==='blocked'||ms.verdict==='baseline_blocked';
 const geoCol=(geo==null||msBlk)?'var(--ink3)':(geo>=1.0?'var(--good)':geo>=0.95?'var(--warn)':'var(--bad)');
 const link=ms.summary_url?`<a href="${ms.summary_url}" onclick="event.stopPropagation()" title="benchmark_summary.md on GitHub">summary ↗</a>`:'';
 const sub=`${ms.n_shapes!=null?ms.n_shapes+' shapes':''}`+
   `${ms.n_flydsl_correct!=null?` · ${ms.n_flydsl_correct} correct`:''}`+
   `${link?' · '+link:''}`;
 return `<div class="msrow" onclick="event.stopPropagation()">
   <div class="msv" style="color:${geoCol}">${msGeo(geo)}</div>
   <div class="msmeta"><span class="mslbl">multishape cold-cache${ms.n_shapes!=null?' ('+ms.n_shapes+' shapes)':''}</span>
     <span class="mssub">${sub}</span></div>
   ${msModelsChip(ms)}
   ${msBadge(ms.verdict)}</div>`;
}
function truncMid(s,n){if(!s||s.length<=n)return s;const a=Math.ceil(n/2)-1,b=Math.floor(n/2)-1;return s.slice(0,a)+'…'+s.slice(s.length-b);}

/* ---- drawer ---- */
function openDrawer(stem){
 const k=K.find(x=>x.stem===stem);if(!k)return;
 const col=accentColor(k.verdict);
 const occStep=k.next_occ_step?`${k.next_occ_step.waves} waves needs VGPR ≤ ${k.next_occ_step.vgpr_budget}`:'';
 let html=`<div class="dhead" style="border-bottom:2px solid ${col}">
   <span class="x" onclick="window.__closeDrawer()">✕</span>
   <h2>${k.name}</h2>
   <div class="kn">${k.jit_kernel||k.test} · ${k.op_category}</div>
   <div style="margin-top:10px">${verdictBadge(k.verdict)} ${k.baseline_name?`<span class="dim mono" style="font-size:12px">vs ${k.baseline_name}</span>`:''}</div>
 </div><div class="dbody">`;

 if(k.headline){html+=`<div class="note" style="border-left-color:${col};margin-top:0;font-size:13.5px">${escapeHTML(k.headline)}</div>`;}

 // headline KV (rocprof single-shape)
 html+=`<h4>rocprof · 1-shape ATT capture</h4><div class="kv">
   <div class="cell"><div class="v fly">${f1(k.flydsl_us)} µs</div><div class="k">FlyDSL latency</div></div>
   <div class="cell"><div class="v">${k.baseline_us?f1(k.baseline_us)+' µs':'—'}</div><div class="k">${shorten(k.baseline_name)||'baseline'}</div></div>
   <div class="cell"><div class="v" style="color:${col}">${fx(k.speedup_vs_baseline)}</div><div class="k">FlyDSL vs baseline (1 shape)</div></div>
   <div class="cell"><div class="v">${k.tflops?f1(k.tflops)+' TF':(k.bandwidth_gbs?f1(k.bandwidth_gbs)+' GB/s':'—')}</div><div class="k">throughput</div></div>
 </div>`;

 // multishape cold-cache (distinct from rocprof single-shape above)
 {
  const ms=k.multishape;
  html+=`<h4>multishape cold-cache (CUDA-graph + L2 flush)</h4>`;
  if(!ms){
   html+=`<div class="note" style="border-left-color:var(--slate)">No multishape benchmark run for this kernel.</div>`;
  }else{
   const _msBlk=ms.verdict==='blocked'||ms.verdict==='baseline_blocked';
   const geoCol=(ms.geomean_vs_best==null||_msBlk)?'var(--ink2)':(ms.geomean_vs_best>=1.0?'var(--good)':ms.geomean_vs_best>=0.95?'var(--warn)':'var(--bad)');
   html+=`<div class="kv">
     <div class="cell"><div class="v" style="color:${geoCol}">${msGeo(ms.geomean_vs_best)}</div><div class="k">geomean vs best correct baseline</div></div>
     <div class="cell"><div class="v">${ms.n_shapes??'—'}</div><div class="k">shapes swept</div></div>
     <div class="cell"><div class="v">${ms.n_flydsl_correct??'—'}${ms.n_shapes!=null?` / ${ms.n_shapes}`:''}</div><div class="k">FlyDSL correct</div></div>
     <div class="cell"><div class="v">${(ms.models&&ms.models.length)||0}</div><div class="k">models covered</div></div>
   </div>
   <div style="margin-top:10px">${msBadge(ms.verdict)} <span class="dim mono" style="font-size:11.5px">verdict · vs_best on ${ms.vs_best_n??'—'} shapes${ms.weighted_geomean!=null?` · weighted ${msGeo(ms.weighted_geomean)}`:''}</span></div>`;
   if(ms.models&&ms.models.length){
    html+=`<div class="tagrow" style="margin-top:10px">`+ms.models.map(m=>`<span class="dlink" style="cursor:default;font-size:11px">${escapeHTML(m)}</span>`).join('')+`</div>`;
   }
   if(ms.summary_url){
    html+=`<div class="dlinks" style="margin-top:10px"><a class="dlink" href="${ms.summary_url}">📊 benchmark_summary.md</a></div>`;
   }
  }
 }
 if(k.extra_baselines&&Object.keys(k.extra_baselines).length){
   html+=`<h4>Baseline matrix (µs)</h4><table class="t"><tr><th>impl</th><th>µs</th></tr>`+
     Object.entries(k.extra_baselines).map(([n,v])=>`<tr><td>${n}</td><td class="mono">${f1(v)}</td></tr>`).join('')+`</table>`;
 }

 if(k.has_bundle){
  // occupancy / regs
  html+=`<h4>Occupancy &amp; register pressure (rocprofv3 ATT)</h4><div class="kv">
    <div class="cell"><div class="v">${k.occupancy} <span style="font-size:12px;color:var(--ink3)">w/SIMD</span></div><div class="k">${occStep||'occupancy'}</div></div>
    <div class="cell"><div class="v">${k.arch_vgpr||'—'}</div><div class="k">arch VGPR</div></div>
    <div class="cell"><div class="v">${k.ins}</div><div class="k">ISA instructions (${k.mapped_pct}% mapped)</div></div>
    <div class="cell"><div class="v">${k.stall_pct_total?Math.round(k.stall_pct_total)+'%':'—'}</div><div class="k">cycles stalled</div></div>
  </div>`;
  // stall taxonomy
  if(k.stall_breakdown&&k.stall_breakdown.length){
   html+=`<h4>Stall taxonomy</h4><div class="minibar">`+
     k.stall_breakdown.filter(s=>s.pct>0).map(s=>`<span style="width:${s.pct}%;background:${sc(s.type)}" title="${s.type}"></span>`).join('')+`</div>`;
   html+=`<table class="t" style="margin-top:8px"><tr><th>stall type</th><th>cycles</th><th>% of stalls</th></tr>`+
     k.stall_breakdown.filter(s=>s.pct>0).map(s=>`<tr><td><span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${sc(s.type)};margin-right:6px"></span>${s.type}</td><td class="mono">${s.stall}</td><td class="mono">${s.pct}%</td></tr>`).join('')+`</table>`;
  }
  // top source lines
  if(k.top_source_lines&&k.top_source_lines.length){
   html+=`<h4>Top hotspot source lines (PC→Python)</h4><table class="t"><tr><th>%</th><th>type</th><th>source : line</th></tr>`+
     k.top_source_lines.slice(0,8).map(s=>`<tr><td class="mono" style="color:${col}">${s.pct_total}%</td><td class="mono dim">${s.domtype}</td><td class="mono" style="font-size:11.5px">${s.source}</td></tr>`).join('')+`</table>`;
  }
  if(k.inst_mix&&Object.keys(k.inst_mix).length){
   html+=`<h4>Instruction mix (sampled CU)</h4><div class="tagrow">`+
     Object.entries(k.inst_mix).map(([n,v])=>`<span class="dlink" style="cursor:default">${n}: ${v}</span>`).join('')+`</div>`;
  }
 }else{
  html+=`<h4>ATT capture</h4><div class="note">${k.status==='compile_fail'?'This kernel currently <b>fails to compile</b> in this build: '+(k.baseline_notes||'flyc.compile() fast-dispatch path error').slice(0,200):'No instruction-level trace ('+(k.capture_error||'n/a')+').'}</div>`;
 }

 // baseline note
 if(k.baseline_notes){
   html+=`<h4>Head-to-head notes</h4><div class="note">${escapeHTML(k.baseline_notes).slice(0,700)}</div>`;
 }
 if(k.top_recommendation){html+=`<h4>#1 optimization</h4><div class="note" style="border-left-color:var(--good)">${escapeHTML(k.top_recommendation)}</div>`;}
 // links (only when a bundle was produced)
 if(k.has_bundle){
   html+=`<h4>Artifacts</h4><div class="dlinks">
     <a class="dlink" href="${k.report_url}">📄 REPORT.md</a>
     <a class="dlink" href="${k.bundle_url}">📦 ATT bundle</a>
     ${k.kernels&&k.kernels.length?`<span class="dlink" style="cursor:default">kernels/${k.kernels[0]}.py</span>`:''}
   </div>`;
 }
 html+=`</div>`;
 $('#drawer-content').innerHTML=html;
 $('#drawer').classList.add('on');$('#scrim').classList.add('on');
}
function escapeHTML(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
window.__closeDrawer=()=>{$('#drawer').classList.remove('on');$('#scrim').classList.remove('on');};
$('#scrim').onclick=window.__closeDrawer;
document.addEventListener('keydown',e=>{if(e.key==='Escape')window.__closeDrawer();});

/* ---- footer ---- */
function footer(){
 const p=D.provenance;
 $('#foot').innerHTML=`<p>Generated by an 8-GPU rocprofv3 ATT sweep on <span class="mono">${p.gpu}</span> ·
   FlyDSL <span class="mono hl">${p.flydsl_version}</span> (commit <span class="mono">${p.flydsl_commit}</span>) ·
   ROCm <span class="mono">${p.rocm}</span> · ${p.capture_date}.</p>
   <p class="dim3">Each kernel's bundle (ATT trace + counters + REPORT.md) lives under <span class="mono">examples/</span>.
   Baselines: AIter / CK / hipBLASLt at matched shapes. FlyDSL is the subject; baselines are the yardstick.
   Deferred (multi-GPU): ${D.deferred.map(d=>d.test).join(', ')}.</p>`;
}

/* init */
header();speedChart();catPills();renderGrid();footer();
$('#search').addEventListener('input',e=>{q=e.target.value.toLowerCase();renderGrid();});
$('#sort').addEventListener('change',e=>{sortBy=e.target.value;renderGrid();});
$('#msverdict').addEventListener('change',e=>{msFilter=e.target.value;renderGrid();});
})();
