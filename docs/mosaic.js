(() => {
 const hero=document.querySelector('.hero-mosaic');
 if(!hero)return;
 const videos=[...hero.querySelectorAll('video')];
 const reduced=matchMedia('(prefers-reduced-motion: reduce)');
 let enabled=!reduced.matches&&!navigator.connection?.saveData,inView=false,revision=0;
 async function sync(){
  const current=++revision;
  if(!enabled||!inView||document.hidden){videos.forEach(v=>v.pause());return;}
  const results=await Promise.allSettled(videos.map(v=>{if(!v.getAttribute('src'))v.src=v.dataset.src;return v.play();}));
  if(current!==revision)return;
  if(results.every(r=>r.status==='rejected')){enabled=false;}
 }
 reduced.addEventListener('change',()=>{enabled=!reduced.matches;sync();});
 document.addEventListener('visibilitychange',sync);
 new IntersectionObserver(entries=>{inView=entries[0].isIntersecting;sync();},{threshold:0}).observe(hero);
})();
