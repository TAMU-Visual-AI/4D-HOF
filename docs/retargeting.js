import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';

const root=document.querySelector('[data-retargeting]'),embedded=root.dataset.embedded==='true';
const loader=new GLTFLoader(),previews=[];
// Share the small local server fairly between the two passive previews.
let transfers=0;const waiting=[];
async function transfer(task){
 if(transfers>=3)await new Promise(resolve=>waiting.push(resolve));else transfers++;
 try{return await task();}finally{const next=waiting.shift();if(next)next();else transfers--;}
}
async function fetchData(url,type='arrayBuffer'){return transfer(async()=>{const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw Error('Unable to load '+url);return r[type]();});}
function loadModel(url){return transfer(()=>loader.loadAsync(url));}
function dispose(group){
 const geos=new Set(),mats=new Set(),textures=new Set();
 group.traverse(o=>{if(o.geometry)geos.add(o.geometry);if(o.material)(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>mats.add(m));});
 mats.forEach(m=>Object.values(m).filter(x=>x?.isTexture).forEach(t=>textures.add(t)));
 geos.forEach(g=>g.dispose());textures.forEach(t=>t.dispose());mats.forEach(m=>m.dispose());group.clear();
}

class Preview{
 constructor(seq){
  this.seq=seq;this.frame=-1;this.active=false;
  const panel=document.createElement('section');this.panel=panel;panel.id=`preview-${seq}`;panel.hidden=true;panel.className='scene-panel';panel.setAttribute('aria-label',`${seq} retargeting preview`);
  panel.innerHTML='<div class="scene-heading"><h3 class="scene-title"></h3><span>Human + five robot hands</span></div><div class="scene-stage"><div class="scene-labels"></div><div class="scene-status" role="status">Loading animation…</div></div>';
  panel.querySelector('.scene-title').textContent=seq;root.querySelector('[data-retarget-previews]').append(panel);
  this.stage=panel.querySelector('.scene-stage');this.labels=panel.querySelector('.scene-labels');this.message=panel.querySelector('.scene-status');
  this.renderer=new THREE.WebGLRenderer({antialias:true});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));this.renderer.setClearColor('#ffffff');this.renderer.outputColorSpace=THREE.SRGBColorSpace;
  this.renderer.domElement.setAttribute('role','img');this.renderer.domElement.setAttribute('aria-label',`${seq}: looping reconstruction and five retargeted robot hands`);this.stage.prepend(this.renderer.domElement);
  this.scene=new THREE.Scene();this.camera=new THREE.OrthographicCamera(-1,1,1,-1,.001,100);
  this.scene.add(new THREE.HemisphereLight(0xffffff,0x687466,2.5));
  const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(1,2,3);this.scene.add(light);
  this.content=new THREE.Group();this.scene.add(this.content);
  new ResizeObserver(()=>this.layout()).observe(this.stage);
 }
 setFrame(frame){
  const data=this.data;this.frame=frame;this.stage.dataset.frame=frame;
  const vertices=data.vertices.subarray(frame*778*3,(frame+1)*778*3),center=new THREE.Vector3();
  for(let i=0;i<vertices.length;i+=3)center.add(new THREE.Vector3(vertices[i],vertices[i+1],vertices[i+2]));
  center.divideScalar(778);
  data.geometry.attributes.position.array.set(vertices);data.geometry.attributes.position.needsUpdate=true;data.geometry.computeVertexNormals();data.geometry.computeBoundingSphere();
  for(const cell of data.cells){
   // Translate each complete hand/object pair for display, preserving relative motion.
   cell.source.position.set(-center.x,center.y,center.z);
   cell.object.matrix.fromArray(data.objectPoses.subarray(frame*16,frame*16+16)).transpose();cell.object.matrixWorldNeedsUpdate=true;
   if(cell.gridCenter)cell.slot.position.set(cell.gridCenter.x-cell.centers[frame*3],cell.gridCenter.y-cell.centers[frame*3+1],0);
   if(!cell.meta)continue;
   const N=cell.meta.linkNames.length;
   cell.nodes.forEach((node,i)=>{node.matrix.fromArray(cell.linkPoses.subarray((frame*N+i)*16,(frame*N+i+1)*16)).transpose();node.matrixWorldNeedsUpdate=true;});
  }
 }
 measure(){
  const bounds=new THREE.Box3();
  for(const cell of this.data.cells){cell.maxSize=new THREE.Vector3();cell.centers=new Float32Array(this.data.meta.frameCount*3);}
  // Measure the entire loop once, so a tight, fixed frame still contains every pose.
  for(let frame=0;frame<this.data.meta.frameCount;frame++){
   this.setFrame(frame);this.content.updateWorldMatrix(true,true);
   for(const cell of this.data.cells){
    const frameBounds=new THREE.Box3(),roots=[cell.hand],origin=cell.slot.position,box=new THREE.Box3().setFromObject(cell.object);
    if(box.getCenter(new THREE.Vector3()).distanceTo(origin)<.5)roots.push(cell.object);
    for(const root of roots)root.traverseVisible(o=>{
     if(!o.isMesh)return;
     if(o.geometry===this.data.geometry||!o.geometry.boundingBox)o.geometry.computeBoundingBox();
     const b=o.geometry.boundingBox.clone().applyMatrix4(o.matrixWorld);b.min.sub(origin);b.max.sub(origin);bounds.union(b);frameBounds.union(b);
    });
    cell.maxSize.max(frameBounds.getSize(new THREE.Vector3()));frameBounds.getCenter(new THREE.Vector3()).toArray(cell.centers,frame*3);
   }
  }
  this.bounds=bounds;
 }
 layout(){
  const w=this.stage.clientWidth,h=this.stage.clientHeight;if(!w||!h)return;
  this.renderer.setSize(w,h,false);
  if(!this.data)return;
  const columns=w<600?2:3,rows=Math.ceil(this.data.cells.length/columns),cellWidth=w/columns,cellHeight=h/rows;
  const size=new THREE.Vector3();
  for(const cell of this.data.cells)size.max(cell.maxSize);
  const labelHeight=38,gutter=10;
  const scale=Math.max(size.x/(cellWidth-2*gutter),size.y/(cellHeight-labelHeight-2*gutter));
  this.camera.left=-w*scale/2;this.camera.right=w*scale/2;this.camera.top=h*scale/2;this.camera.bottom=-h*scale/2;
  this.camera.position.set(0,0,this.bounds.max.z+1);this.camera.lookAt(0,0,0);this.camera.updateProjectionMatrix();
  this.data.cells.forEach((cell,i)=>{
   const x=(i%columns+.5)*cellWidth,y=(Math.floor(i/columns)+.5)*cellHeight+labelHeight/2;
   cell.gridCenter=new THREE.Vector2((x-w/2)*scale,(h/2-y)*scale);
   cell.slot.position.set(cell.gridCenter.x-cell.centers[this.frame*3],cell.gridCenter.y-cell.centers[this.frame*3+1],0);
   cell.label.style.left=`${x}px`;cell.label.style.top=`${Math.floor(i/columns)*cellHeight+6}px`;
  });
  this.renderer.render(this.scene,this.camera);
 }
 async load(robots){
  const acquired=new THREE.Group(),src=`assets/interactive/${this.seq}/`;
  try{
   const requests=[fetchData(src+'hand-vertices.bin'),fetchData(src+'hand-faces.bin'),fetchData(src+'object-poses.bin'),loadModel(src+'object.glb').then(o=>{acquired.add(o.scene);return o;})];
   const robotRequests=robots.map(async spec=>{
    const base=`assets/retargeting/${spec.prefix}${this.seq}/`;
    const loaded=await Promise.allSettled([fetchData(base+'metadata.json','json'),fetchData(base+'link-poses.bin'),loadModel(base+'robot.glb').then(o=>{acquired.add(o.scene);return o;})]);
    const failed=loaded.find(r=>r.status==='rejected');if(failed)throw failed.reason;
    const [meta,links,model]=loaded.map(r=>r.value);return {spec,meta,linkPoses:new Float32Array(links),model:model.scene};
   });
   const results=await Promise.allSettled([...requests,...robotRequests]);
   const failure=results.find(r=>r.status==='rejected');if(failure)throw failure.reason;
   const [vertices,faces,poses,object,...hands]=results.map(r=>r.value),meta=hands[0].meta;
   if(hands.some(h=>h.meta.frameCount!==meta.frameCount||h.meta.fps!==meta.fps))throw Error('Hand trajectories have incompatible timing');
   const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(778*3),3));geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces),1));
   const human=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:'#ce997b',roughness:.68,side:THREE.DoubleSide}));
   object.scene.traverse(o=>{if(o.isMesh){if(!o.geometry.attributes.normal)o.geometry.computeVertexNormals();for(const m of Array.isArray(o.material)?o.material:[o.material]){m.side=THREE.DoubleSide;m.metalness=0;m.needsUpdate=true;}}});
   const cells=[];
   for(const entry of [null,...hands]){
    const slot=new THREE.Group(),source=new THREE.Group();source.rotation.x=Math.PI;slot.add(source);this.content.add(slot);
    const copy=object.scene.clone(true);copy.matrixAutoUpdate=false;source.add(copy);
    const label=document.createElement('div');label.className='scene-label'+(entry?'':' human-label');
    const title=document.createElement('strong');title.textContent=entry?entry.spec.name:'Human hand';
    const subtitle=document.createElement('span');subtitle.textContent=entry?`${entry.spec.fingers} fingers`:'Reconstruction';label.append(title,subtitle);this.labels.append(label);
    const cell={slot,source,object:copy,label,hand:entry?entry.model:human};source.add(cell.hand);
    if(entry){
     Object.assign(cell,entry);
     cell.nodes=entry.meta.linkNames.map((name,j)=>{const node=entry.model.getObjectByName(entry.meta.visualNodeNames?.[j]||name);if(!node)throw Error('Missing robot link '+name);node.matrixAutoUpdate=false;if(name==='forearm')node.visible=false;return node;});
    }
    cells.push(cell);
   }
   acquired.clear();this.data={meta,vertices:new Float32Array(vertices),objectPoses:new Float32Array(poses),geometry,cells};
   this.measure();this.setFrame(0);this.layout();this.start=performance.now();
   this.message.textContent='';
  }catch(e){dispose(acquired);dispose(this.content);this.data=null;this.labels.replaceChildren();this.message.textContent=`Unable to load ${this.seq}: ${e.message}`;console.error(e);}
 }
 animate(now){
  if(!this.active||!this.data||document.hidden)return;
  const frame=Math.floor((now-this.start)*this.data.meta.fps/1000)%this.data.meta.frameCount;
  if(frame===this.frame)return;
  this.setFrame(frame);this.renderer.render(this.scene,this.camera);
 }
}

// Keep both cases ready; the thumbnail picker selects one automatic preview.
const url=new URL(location.href),requestedSequence=url.searchParams.get(embedded?'retarget':'seq');
if(!embedded)for(const key of ['robot','policy','physics'])url.searchParams.delete(key);
const picker=root.querySelector('[data-retarget-picker]');
function choose(seq){
 for(const preview of previews){
  preview.active=preview.seq===seq;preview.panel.hidden=!preview.active;
  if(preview.active){
   if(preview.data){preview.setFrame(0);preview.start=performance.now();}
   preview.layout();
  }
 }
 for(const button of picker.querySelectorAll('button'))button.setAttribute('aria-pressed',String(button.dataset.seq===seq));
 if(!embedded){url.searchParams.set('seq',seq);history.replaceState(null,'',url.pathname+url.search+url.hash);}
}
function animate(now){requestAnimationFrame(animate);for(const preview of previews)preview.animate(now);}
// Load section 05 as it approaches the viewport; once started it stays active.
if(embedded&&'IntersectionObserver' in window){
 await new Promise(resolve=>{
  const observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting)){observer.disconnect();resolve();}},{rootMargin:'800px'});
  observer.observe(root);
 });
}
requestAnimationFrame(animate);
try{
 const [sequences,robots]=await Promise.all([fetchData('assets/retargeting/sequences.json','json'),fetchData('assets/retargeting/robots.json','json')]);
 for(const seq of sequences){
  previews.push(new Preview(seq));
  const button=document.createElement('button');button.type='button';button.className='sequence-card';button.dataset.seq=seq;
  button.setAttribute('aria-label',`Show ${seq}`);button.setAttribute('aria-controls',`preview-${seq}`);button.setAttribute('aria-pressed','false');
  const image=document.createElement('img');image.src=`assets/interactive/${seq}/thumbnail.jpg`;image.alt='';image.width=120;image.height=90;
  button.append(image);button.onclick=()=>choose(seq);picker.append(button);
 }
 picker.addEventListener('keydown',event=>{
  const button=event.target.closest('button[data-seq]');if(!button)return;
  const index=sequences.indexOf(button.dataset.seq),target={ArrowLeft:index-1,ArrowRight:index+1,Home:0,End:sequences.length-1}[event.key];
  if(target===undefined)return;event.preventDefault();
  const seq=sequences[Math.max(0,Math.min(sequences.length-1,target))];choose(seq);picker.querySelector(`[data-seq="${seq}"]`).focus({preventScroll:true});
 });
 choose(sequences.includes(requestedSequence)?requestedSequence:sequences[0]);
 await Promise.all(previews.map(preview=>preview.load(robots)));
}catch(e){root.querySelector('[data-retarget-status]').textContent=e.message;console.error(e);}
