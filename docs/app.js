'use strict';
const examples=[{id:'ABF14',group:'long'},{id:'GPMF14',group:'long'},{id:'SM4',group:'long'},{id:'MC4',group:'short'},{id:'MDF14',group:'short'},{id:'ShSu10',group:'short'}];
const methods=[{key:'GT',file:'gt',label:'GT'},{key:'ours',file:'ours',label:'4D-HOF'},{key:'HOLD',file:'hold',label:'HOLD'},{key:'MagicHOI',file:'magichoi',label:'MagicHOI'},{key:'CHOIR',file:'choir',label:'CHOIR'}];
const $=s=>document.querySelector(s);
let sequence=examples[0].id,view='reconstruction',videos=[],duration=0,playing=false,revision=0,ready=false,visible=false,lastTick=0;
let wantsPlayback=!matchMedia('(prefers-reduced-motion: reduce)').matches;
function playbackUI(){ $('#play-label').textContent=playing?'Pause':'Play';$('#play-icon').textContent=playing?'Ⅱ':'▶';$('#play-button').setAttribute('aria-label',`${playing?'Pause':'Play'} all videos`); }
function pause(){videos.forEach(v=>v.pause());playing=false;playbackUI();}
async function play(){if(!ready||!videos.length)return;const current=revision;playing=true;playbackUI();const results=await Promise.allSettled(videos.map(v=>v.play()));if(current!==revision)return;if(results.some(r=>r.status==='rejected')){pause();wantsPlayback=false;$('#viewer-status').textContent='Select Play to start the comparison.';}}
function seek(t){videos.forEach(v=>{if(v.readyState>=1)v.currentTime=Math.min(t,v.duration);});updateTime(t);}
function updateTime(t){$('#timeline').value=duration?Math.round(t/duration*1000):0;$('#timeline').setAttribute('aria-valuetext',`${t.toFixed(1)} of ${duration.toFixed(1)} seconds`);}
function renderExamples(){
 $('#example-selector').innerHTML=examples.map(({id,group})=>`<button type="button" class="example-card" data-example="${id}" aria-pressed="${id===sequence}" aria-label="Select sequence ${id}"><img src="assets/posters/${group}-${id}-rgb.jpg" alt="" width="480" height="360"><span><strong>${id}</strong></span><span class="example-check" aria-hidden="true">${id===sequence?'✓':''}</span></button>`).join('');
 $('#selection-count').textContent=`EXAMPLE ${String(examples.findIndex(example=>example.id===sequence)+1).padStart(2,'0')} / ${String(examples.length).padStart(2,'0')}`;
}
function loadVideo(v){return new Promise(resolve=>{let settled=false;const finish=ok=>{if(settled)return;settled=true;clearTimeout(timer);resolve(ok);};const timer=setTimeout(()=>finish(false),25000);v.addEventListener('loadeddata',()=>finish(true),{once:true});v.addEventListener('error',()=>finish(false),{once:true});v.load();});}
async function loadComparison(){
 const group=examples.find(example=>example.id===sequence).group;
 const current=++revision;pause();ready=false;duration=0;updateTime(0);
 ['#play-button','#restart-button','#timeline'].forEach(s=>$(s).disabled=true);
 $('#viewer-status').textContent='Loading comparison…';$('#sequence-title').textContent=sequence;
 document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.view===view));renderExamples();
 videos.forEach(v=>{v.removeAttribute('src');v.load();});
 $('#video-grid').innerHTML=methods.map(m=>{
 const missing=group==='long'&&sequence==='ABF14'&&m.key==='MagicHOI';
 const folder=m.key==='CHOIR'&&group==='short'?'choir':m.key;
 const src=`assets/${group}/${folder}/${sequence}/${m.file}${view==='overlay'?'_overlay':''}.mp4`;
 return `<figure class="video-card ${m.key==='ours'?'ours':''}"><div class="video-stage">${missing?'<div class="unavailable"><span aria-hidden="true">—</span>Not available<br>for this sequence</div>':`<video src="${src}" muted playsinline preload="auto" aria-label="${m.label} ${view} for ${sequence}"></video>`}</div><figcaption><span>${m.label}</span>${m.key==='ours'?'<span class="ours-badge">Ours</span>':''}</figcaption></figure>`;
 }).join('');
 videos=Array.from(document.querySelectorAll('#video-grid video'));const loadingVideos=[...videos];
 loadingVideos.forEach(v=>{v.muted=true;v.playbackRate=Number($('#speed').value);});
 const loaded=await Promise.all(loadingVideos.map(loadVideo));if(current!==revision)return;
 loaded.forEach((ok,i)=>{if(!ok){const v=loadingVideos[i];v.pause();v.parentElement.innerHTML='<div class="unavailable">Video could not load.<br>Reselect the example to retry.</div>';}});
 videos=loadingVideos.filter((v,i)=>loaded[i]);
 if(!videos.length){$('#viewer-status').textContent='Videos could not load. Reselect an example to retry.';return;}
 duration=Math.min(...videos.map(v=>v.duration));ready=true;
 ['#play-button','#restart-button','#timeline'].forEach(s=>$(s).disabled=false);
 $('#viewer-status').textContent=`${videos.length} methods · Shared playback${loaded.some(ok=>!ok)?' · Some videos unavailable':''}`;
 videos.forEach(v=>v.addEventListener('ended',()=>{if(current!==revision||!playing)return;seek(0);if(wantsPlayback&&visible)play();}));
 if(wantsPlayback&&visible)play();
}
$('#example-selector').addEventListener('click',e=>{const b=e.target.closest('[data-example]');if(!b)return;sequence=b.dataset.example;loadComparison();});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{if(view===b.dataset.view)return;view=b.dataset.view;loadComparison();}));
document.querySelectorAll('[data-open-sequence]').forEach(b=>b.addEventListener('click',()=>{sequence=b.dataset.openSequence;loadComparison();}));
$('#play-button').addEventListener('click',()=>{wantsPlayback=!playing;if(playing)pause();else play();});
$('#restart-button').addEventListener('click',()=>seek(0));
$('#timeline').addEventListener('input',e=>seek(Number(e.target.value)/1000*duration));
$('#speed').addEventListener('change',e=>videos.forEach(v=>v.playbackRate=Number(e.target.value)));
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible)pause();else if(wantsPlayback&&ready)play();},{threshold:.12}).observe($('#video-grid'));
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();else if(visible&&wantsPlayback)play();});
function tick(now){if(playing&&videos.length&&now-lastTick>150){lastTick=now;const t=videos[0].currentTime;if(t>=duration-.06){seek(0);}else{videos.slice(1).forEach(v=>{if(Math.abs(v.currentTime-t)>.12)v.currentTime=t;});updateTime(t);}}requestAnimationFrame(tick);}
loadComparison();requestAnimationFrame(tick);
