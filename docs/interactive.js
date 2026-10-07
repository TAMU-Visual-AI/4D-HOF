import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

const root=document.querySelector('#interactive-demo');
const el=id=>root.querySelector(`#${id}`);
const video=el('demo-rgb');
const rgbCanvas=el('demo-input-canvas');
const rgbContext=rgbCanvas.getContext('2d');
const stage=el('demo-stage');
const status=el('demo-status');
function setStatus(message){status.textContent=message;status.hidden=!message;}
let meta,renderer,scene,camera,controls,hand,object,positions,poses,videoTexture,cv;
let frame=0,mode='orbit',ready=false,visible=true,scrubbing=false,wantedPlayback=false;
let loadVersion=0,blobURL=null,activeSequence='ABF14',materials=[],examples=[],selectorLoading=true;
const target=new THREE.Vector3();
async function fetchData(url,kind='arrayBuffer'){
 const r=await fetch(url);if(!r.ok)throw Error(`${url} unavailable`);return r[kind]();
}
function updateExampleNavigation(){
 const index=examples.indexOf(activeSequence);
 el('demo-example-prev').disabled=selectorLoading||index<=0;
 el('demo-example-next').disabled=selectorLoading||index>=examples.length-1;
}
function renderExamples(){
 const fragment=document.createDocumentFragment();
 for(const id of examples){
  const button=document.createElement('button');button.type='button';button.className='demo-example-card';button.dataset.demoSequence=id;button.disabled=true;
  button.setAttribute('aria-label',`Show interactive sequence ${id}`);button.setAttribute('aria-pressed',String(id===activeSequence));
  const image=document.createElement('img');image.src=`assets/interactive/${id}/thumbnail.jpg${['07','10'].includes(id)?'?v=20261007':''}`;image.alt='';image.width=480;image.height=360;image.loading='lazy';
  button.append(image);fragment.append(button);
 }
 el('demo-examples').replaceChildren(fragment);
}
function chooseExample(index){
 if(selectorLoading||index<0||index>=examples.length)return;
 const id=examples[index];
 const button=root.querySelector(`[data-demo-sequence="${id}"]`);
 button.focus({preventScroll:true});button.scrollIntoView({block:'nearest',inline:'nearest'});
 if(id!==activeSequence||!ready)loadSequence(id);
}
function setFrame(index){
 if(!ready)return;
 frame=Math.max(0,Math.min(meta.frameCount-1,index));
 hand.geometry.attributes.position.array.set(positions.subarray(frame*meta.vertexCount*3,(frame+1)*meta.vertexCount*3));
 hand.geometry.attributes.position.needsUpdate=true;hand.geometry.computeVertexNormals();
 object.matrix.fromArray(poses.subarray(frame*16,frame*16+16)).transpose();object.matrixWorldNeedsUpdate=true;
 el('demo-frame').value=frame;el('demo-frame').setAttribute('aria-valuetext',`Frame ${frame+1} of ${meta.frameCount}`);
}
function playbackUI(){el('demo-play').textContent=video.paused?'Play':'Pause';el('demo-play').setAttribute('aria-label',video.paused?'Play interactive sequence':'Pause interactive sequence');}
async function play(){if(!ready)return;try{await video.play();}catch{wantedPlayback=false;setStatus('Select Play to start the sequence.');}playbackUI();}
function pause(){video.pause();playbackUI();}
function seek(index){if(!ready)return;pause();wantedPlayback=false;setFrame(index);video.currentTime=(frame+0.25)/meta.fps;}
function resetOrbit(){
 hand.geometry.computeBoundingBox();cv.updateMatrixWorld(true);
 const box=new THREE.Box3().setFromObject(cv);box.getCenter(target);
 const size=box.getSize(new THREE.Vector3());
 const distance=Math.max(size.x,size.y,size.z)/Math.tan(THREE.MathUtils.degToRad(45/2))*.85;
 controls.target.copy(target);camera.position.copy(target).add(new THREE.Vector3(.55,.25,1).normalize().multiplyScalar(distance));camera.up.set(0,1,0);camera.lookAt(target);
 camera.fov=45;camera.aspect=meta.width/meta.height;camera.updateProjectionMatrix();controls.update();
}
function cameraProjection(){
 const [[fx,,cx],[,fy,cy]]=meta.intrinsics,w=meta.width,h=meta.height,n=.005,f=20;
 camera.projectionMatrix.set(2*fx/w,0,1-2*cx/w,0,0,2*fy/h,2*cy/h-1,0,0,0,-(f+n)/(f-n),-2*f*n/(f-n),0,0,-1,0);
 camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}
function setMode(next){
 if(!ready)return;mode=next;
 root.querySelectorAll('[data-demo-view]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.demoView===mode));
 controls.enabled=mode==='orbit';scene.background=mode==='overlay'?videoTexture:new THREE.Color('#f5f8f6');
 if(mode==='overlay'){camera.position.set(0,0,0);camera.up.set(0,1,0);camera.lookAt(0,0,-1);cameraProjection();}else resetOrbit();
 el('demo-hint').textContent=mode==='orbit'?'Drag to rotate · Scroll or pinch to zoom':'Calibrated camera view · Adjust mesh opacity to inspect alignment';
 stage.setAttribute('aria-label',mode==='orbit'?'Interactive 3D hand-object reconstruction. Drag to rotate, scroll to zoom.':'Hand-object reconstruction overlaid on the RGB video.');
}
function resize(){if(renderer)renderer.setSize(stage.clientWidth,stage.clientHeight,false);}
function updateOpacity(){const opacity=Number(el('demo-opacity').value)/100;el('demo-opacity-value').textContent=`${Math.round(opacity*100)}%`;materials.forEach(m=>{m.opacity=opacity;m.transparent=opacity<1;m.needsUpdate=true;});}
function disposeModel(model){
 if(!model)return;
 const geometries=new Set(),mats=new Set(),textures=new Set();
 model.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])mats.add(m);});
 for(const m of mats){for(const value of Object.values(m))if(value?.isTexture)textures.add(value);m.dispose();}
 textures.forEach(t=>t.dispose());geometries.forEach(g=>g.dispose());
}
function clearSequence(){
 if(cv){scene.remove(cv);disposeModel(cv);cv=null;}
 materials=[];hand=null;object=null;positions=null;poses=null;
 video.removeAttribute('src');video.load();if(blobURL){URL.revokeObjectURL(blobURL);blobURL=null;}
 rgbContext.clearRect(0,0,rgbCanvas.width,rgbCanvas.height);
 if(renderer){scene.background=new THREE.Color('#f5f8f6');renderer.render(scene,camera);}
}
async function loadSequence(sequence){
 const version=++loadVersion;activeSequence=sequence;selectorLoading=true;updateExampleNavigation();ready=false;scrubbing=false;wantedPlayback=false;pause();clearSequence();
 root.querySelectorAll('[data-demo-control], [data-demo-sequence]').forEach(x=>x.disabled=true);
 root.querySelectorAll('[data-demo-sequence]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.demoSequence===sequence));
 el('demo-frame').value=0;el('demo-frame').setAttribute('aria-valuetext','Loading sequence');
 el('demo-loading').hidden=false;el('demo-loading').textContent=`Loading ${sequence}…`;setStatus('');
 const base=`assets/interactive/${sequence}/`;let pendingModel;
 try{
  const results=await Promise.allSettled([
   fetchData(base+'metadata.json','json'),fetchData(base+'hand-vertices.bin'),fetchData(base+'hand-faces.bin'),fetchData(base+'object-poses.bin'),
   new GLTFLoader().loadAsync(base+'object.glb'),fetchData(base+'rgb.mp4'+(['07','10'].includes(sequence)?'?v=20261007':''))
  ]);
  pendingModel=results[4].status==='fulfilled'?results[4].value.scene:null;
  if(version!==loadVersion){disposeModel(pendingModel);return;}
  const failure=results.find(r=>r.status==='rejected');if(failure)throw failure.reason;
  const [data,vb,fb,pb,gltf,rgb]=results.map(r=>r.value);meta=data;
  positions=new Float32Array(vb);poses=new Float32Array(pb);
  if(positions.length!==meta.frameCount*meta.vertexCount*3||poses.length!==meta.frameCount*16)throw Error('Geometry frame count mismatch');
  // Blob URLs support exact seeking on static servers without HTTP byte ranges.
  blobURL=URL.createObjectURL(new Blob([rgb],{type:'video/mp4'}));
  await new Promise((resolve,reject)=>{
   const done=()=>{cleanup();resolve();},error=()=>{cleanup();reject(Error('RGB video unavailable'));};
   const timeout=setTimeout(()=>{cleanup();reject(Error('RGB video timed out'));},20000);
   const cleanup=()=>{clearTimeout(timeout);video.removeEventListener('loadeddata',done);video.removeEventListener('error',error);};
   video.addEventListener('loadeddata',done);video.addEventListener('error',error);video.src=blobURL;video.load();
  });
  if(version!==loadVersion){disposeModel(pendingModel);return;}
  video.playbackRate=Number(el('demo-speed').value);
  rgbCanvas.width=meta.width;rgbCanvas.height=meta.height;rgbCanvas.setAttribute('aria-label',`${sequence} input video frame`);
  stage.style.aspectRatio=rgbCanvas.style.aspectRatio=`${meta.width} / ${meta.height}`;
  cv=new THREE.Group();cv.scale.set(1,-1,-1);scene.add(cv);
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(meta.vertexCount*3),3).setUsage(THREE.DynamicDrawUsage));geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(fb),1));
  const handMaterial=new THREE.MeshStandardMaterial({color:0xc4b8df,roughness:.75,metalness:0,side:THREE.DoubleSide});materials.push(handMaterial);
  hand=new THREE.Mesh(geometry,handMaterial);hand.frustumCulled=false;cv.add(hand);
  object=new THREE.Group();object.matrixAutoUpdate=false;object.add(gltf.scene);cv.add(object);pendingModel=null;
  gltf.scene.traverse(o=>{if(o.isMesh)for(const m of Array.isArray(o.material)?o.material:[o.material]){m.side=THREE.DoubleSide;materials.push(m);}});
  ready=true;el('demo-frame').max=meta.frameCount-1;setFrame(0);setMode(mode);updateOpacity();resize();
  rgbContext.drawImage(video,0,0,rgbCanvas.width,rgbCanvas.height);
  root.querySelectorAll('[data-demo-control]').forEach(x=>x.disabled=false);el('demo-loading').hidden=true;
 }catch(error){
  disposeModel(pendingModel);console.error('Interactive demo:',error);
  setStatus(`Could not load ${sequence}. Select a sequence to retry.`);el('demo-loading').textContent='3D preview unavailable';
 }finally{if(version===loadVersion){selectorLoading=false;root.querySelectorAll('[data-demo-sequence]').forEach(b=>b.disabled=false);updateExampleNavigation();}}
}
async function init(){
 try{
  renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;
  stage.prepend(renderer.domElement);renderer.domElement.setAttribute('aria-hidden','true');
  scene=new THREE.Scene();scene.background=new THREE.Color('#f5f8f6');camera=new THREE.PerspectiveCamera(45,4/3,.005,20);
  controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;controls.minDistance=.07;controls.maxDistance=2;controls.enablePan=true;
  scene.add(new THREE.HemisphereLight(0xffffff,0x728076,2.5));const key=new THREE.DirectionalLight(0xffffff,2);key.position.set(1,2,2);scene.add(key);
  videoTexture=new THREE.VideoTexture(video);videoTexture.colorSpace=THREE.SRGBColorSpace;resize();new ResizeObserver(resize).observe(stage);
  if('requestVideoFrameCallback' in video){const sync=(_,info)=>{if(ready&&!scrubbing)setFrame(Math.round(info.mediaTime*meta.fps));video.requestVideoFrameCallback(sync);};video.requestVideoFrameCallback(sync);}
  video.addEventListener('seeked',()=>{if(ready&&!scrubbing)setFrame(Math.min(meta.frameCount-1,Math.floor(video.currentTime*meta.fps)));});
  function render(){requestAnimationFrame(render);if(!ready||!visible||document.hidden)return;if(!('requestVideoFrameCallback' in video)&&!video.paused)setFrame(Math.floor(video.currentTime*meta.fps));if(mode==='orbit')controls.update();if(video.readyState>=2)rgbContext.drawImage(video,0,0,rgbCanvas.width,rgbCanvas.height);renderer.render(scene,camera);}render();
  examples=await fetchData('assets/interactive/sequences.json','json');
  if(!Array.isArray(examples)||!examples.length||examples.some(id=>typeof id!=='string'||!/^[A-Za-z0-9_-]+$/.test(id))||new Set(examples).size!==examples.length)throw Error('Invalid sequence list');
  activeSequence=examples[0];renderExamples();loadSequence(activeSequence);
 }catch(error){console.error('Interactive demo:',error);setStatus('The interactive demo could not load. Please reload to try again.');el('demo-loading').textContent='3D preview unavailable';}
}
el('demo-play').addEventListener('click',()=>{wantedPlayback=video.paused;if(video.paused)play();else pause();});
el('demo-frame').addEventListener('input',e=>{scrubbing=true;seek(Number(e.target.value));});el('demo-frame').addEventListener('change',()=>{scrubbing=false;});
el('demo-prev').addEventListener('click',()=>seek(frame-1));el('demo-next').addEventListener('click',()=>seek(frame+1));
el('demo-reset').addEventListener('click',()=>setMode(mode));el('demo-speed').addEventListener('change',e=>video.playbackRate=Number(e.target.value));el('demo-opacity').addEventListener('input',updateOpacity);
root.querySelectorAll('[data-demo-view]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.demoView)));
el('demo-examples').addEventListener('click',event=>{const button=event.target.closest('[data-demo-sequence]');if(button&&!button.disabled)chooseExample(examples.indexOf(button.dataset.demoSequence));});
el('demo-examples').addEventListener('keydown',event=>{
 const button=event.target.closest('[data-demo-sequence]');if(!button||selectorLoading)return;
 const index=examples.indexOf(button.dataset.demoSequence);
 const next={ArrowLeft:index-1,ArrowRight:index+1,Home:0,End:examples.length-1}[event.key];
 if(next!==undefined){event.preventDefault();chooseExample(Math.max(0,Math.min(examples.length-1,next)));}
});
el('demo-example-prev').addEventListener('click',()=>chooseExample(examples.indexOf(activeSequence)-1));
el('demo-example-next').addEventListener('click',()=>chooseExample(examples.indexOf(activeSequence)+1));
stage.addEventListener('dblclick',()=>setMode(mode));video.addEventListener('play',playbackUI);video.addEventListener('pause',playbackUI);
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;if(!visible)pause();else if(wantedPlayback&&ready&&!document.hidden)play();},{threshold:.05}).observe(root);
document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();else if(visible&&wantedPlayback)play();});
window.addEventListener('pagehide',()=>{if(blobURL)URL.revokeObjectURL(blobURL);});
init();
