(() => {
 const root=document.querySelector('#gpt-comparison');if(!root)return;
 const el=id=>root.querySelector(`#${id}`),ids=['ABF12','SM2'];
 let sequence=ids[0],videos=[],duration=0,ready=false,visible=false,playing=false,started=false,version=0,last=0,scrubbing=false,resumeAfterScrub=false;
 let blobURLs=[],controller=null;
 let wanted=!matchMedia('(prefers-reduced-motion: reduce)').matches;
 const controls=[el('gpt-play'),el('gpt-restart'),el('gpt-timeline')];
 function ui(){el('gpt-play-label').textContent=playing?'Pause':'Play';el('gpt-play-icon').textContent=playing?'Ⅱ':'▶';el('gpt-play').setAttribute('aria-label',`${playing?'Pause':'Play'} GPT comparison`);}
 function pause(){videos.forEach(v=>v.pause());playing=false;ui();}
 function progress(t){el('gpt-timeline').value=duration?Math.round(t/duration*1000):0;el('gpt-timeline').setAttribute('aria-valuetext',`${Math.round(duration?t/duration*100:0)} percent`);}
 function seek(t){videos.forEach(v=>{if(v.readyState>=1)v.currentTime=Math.min(t,v.duration-.001);});progress(t);}
 async function play(){if(!ready||!visible||document.hidden)return;const current=version;playing=true;ui();const results=await Promise.allSettled(videos.map(v=>v.play()));if(current!==version)return;if(results.some(r=>r.status==='rejected')){pause();wanted=false;el('gpt-status').textContent='Select Play to start the comparison.';}}
 function loadVideo(v){return new Promise(resolve=>{let timer;const finish=ok=>{clearTimeout(timer);v.removeEventListener('loadeddata',done);v.removeEventListener('error',fail);resolve(ok);};const done=()=>finish(true),fail=()=>finish(false);v.addEventListener('loadeddata',done);v.addEventListener('error',fail);timer=setTimeout(fail,25000);v.load();});}
 async function load(){
  const current=++version;pause();controller?.abort();controller=new AbortController();const signal=controller.signal;blobURLs.forEach(url=>URL.revokeObjectURL(url));blobURLs=[];ready=false;scrubbing=false;duration=0;controls.forEach(c=>c.disabled=true);progress(0);
  el('gpt-sequence-title').textContent=sequence;el('gpt-status').textContent='Loading comparison…';
  root.querySelectorAll('[data-gpt-sequence]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.gptSequence===sequence)));
  videos.forEach(v=>{v.removeAttribute('src');v.load();});
  el('gpt-video-grid').innerHTML=[{id:'gpt',label:'GPT-6 Astra Ultra'},{id:'ours',label:'4D-HOF'}].map(m=>`<figure class="video-card ${m.id==='ours'?'ours':''}"><div class="video-stage"><video data-src="assets/gpt/${sequence}/${m.id}.mp4" poster="assets/gpt/${sequence}/${m.id}.jpg" muted playsinline preload="auto" aria-label="${m.label} overlay for ${sequence}"></video></div><figcaption><span>${m.label}</span>${m.id==='ours'?'<span class="ours-badge">Ours</span>':''}</figcaption></figure>`).join('');
  videos=[...el('gpt-video-grid').querySelectorAll('video')];const pending=[...videos];pending.forEach(v=>{v.muted=true;v.playbackRate=Number(el('gpt-speed').value);});
  let loaded;try{loaded=await Promise.all(pending.map(async v=>{const response=await fetch(v.dataset.src,{signal});if(!response.ok)throw Error('Video unavailable');const blob=await response.blob();if(current!==version)return false;const url=URL.createObjectURL(blob);blobURLs.push(url);v.src=url;return loadVideo(v);}));}catch(error){if(current===version)el('gpt-status').textContent='A video could not load. Select the example to retry.';return;}if(current!==version)return;
  if(loaded.some(ok=>!ok)){el('gpt-status').textContent='A video could not load. Select the example to retry.';return;}
  duration=Math.min(...videos.map(v=>v.duration));ready=true;controls.forEach(c=>c.disabled=false);el('gpt-status').textContent='2 methods · Shared playback';
  videos.forEach(v=>v.addEventListener('ended',()=>{if(current===version&&playing){seek(0);if(wanted)play();}}));
  if(wanted)play();
 }
 function choose(id){if(!ids.includes(id))return;sequence=id;started=true;load();}
 el('gpt-examples').addEventListener('click',e=>{const b=e.target.closest('[data-gpt-sequence]');if(b)choose(b.dataset.gptSequence);});
 el('gpt-examples').addEventListener('keydown',e=>{const b=e.target.closest('[data-gpt-sequence]');if(!b)return;const index=ids.indexOf(b.dataset.gptSequence);const target={ArrowLeft:index-1,ArrowRight:index+1,Home:0,End:ids.length-1}[e.key];if(target===undefined)return;e.preventDefault();const id=ids[Math.max(0,Math.min(ids.length-1,target))];root.querySelector(`[data-gpt-sequence="${id}"]`).focus({preventScroll:true});choose(id);});
 el('gpt-play').addEventListener('click',()=>{wanted=!playing;if(playing)pause();else play();});
 el('gpt-restart').addEventListener('click',()=>seek(0));
 el('gpt-timeline').addEventListener('input',e=>{if(!scrubbing){resumeAfterScrub=wanted;scrubbing=true;pause();}seek(Number(e.target.value)/1000*duration);});
 el('gpt-timeline').addEventListener('change',()=>{scrubbing=false;if(resumeAfterScrub&&wanted)play();});
 el('gpt-speed').addEventListener('change',()=>videos.forEach(v=>v.playbackRate=Number(el('gpt-speed').value)));
 new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(visible&&!started){started=true;load();}else if(!visible)pause();else if(wanted&&ready)play();},{threshold:.05}).observe(root);
 window.addEventListener('pagehide',()=>{controller?.abort();blobURLs.forEach(url=>URL.revokeObjectURL(url));});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();else if(visible&&wanted)play();});
 function tick(now){if(playing&&!scrubbing&&ready&&now-last>120){last=now;const t=videos[0].currentTime;if(t>=duration-.06)seek(0);else{if(Math.abs(videos[1].currentTime-t)>.1)videos[1].currentTime=t;progress(t);}}requestAnimationFrame(tick);}requestAnimationFrame(tick);
})();
