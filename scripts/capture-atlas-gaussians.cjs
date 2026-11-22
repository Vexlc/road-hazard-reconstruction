/** Capture the real WebGL and model-viewer outputs through Chrome's debugging protocol. */
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {spawn} = require("node:child_process");
const WebSocket = require("ws");

const base = "http://127.0.0.1:8765/tools/asset_atlas/";
const out = process.argv[2];
const captureRoad = process.argv.includes("--road");
const sampling=process.argv.includes("--edge-sampling")?"edges":"area";
const ids = process.argv.slice(3).filter(value=>value!=="--road" && value!=="--edge-sampling");
if (!out || !ids.length) throw Error("Usage: node scripts/capture-atlas-gaussians.cjs OUTPUT ID ...");
const sleep = ms => new Promise(resolve=>setTimeout(resolve,ms));

async function capture() {
  fs.mkdirSync(out,{recursive:true});
  const profile = fs.mkdtempSync(path.join(os.tmpdir(),"roadgs-webgl-"));
  const chrome = spawn(process.env.CHROME || "google-chrome",["--headless=new","--no-sandbox","--disable-dev-shm-usage",
    "--use-gl=angle","--use-angle="+(process.env.ATLAS_CHROME_ANGLE || "swiftshader"),
    "--enable-gpu","--enable-unsafe-swiftshader","--remote-debugging-port=0",
    "--user-data-dir="+profile,"about:blank"],{stdio:"ignore"});
  let socket;
  try {
    const portFile=path.join(profile,"DevToolsActivePort");
    for(let i=0;i<100 && !fs.existsSync(portFile);i++) await sleep(100);
    const port=fs.readFileSync(portFile,"utf8").split("\n")[0];
    const tabs=await(await fetch("http://127.0.0.1:"+port+"/json/list")).json();
    socket=new WebSocket(tabs.find(t=>t.type==="page").webSocketDebuggerUrl);
    await new Promise((resolve,reject)=>{socket.once("open",resolve);socket.once("error",reject)});
    const pending=new Map(), errors=[];
    let seq=0;
    socket.on("message",buf=>{
      const d=JSON.parse(buf);
      if(pending.has(d.id)){pending.get(d.id)(d);pending.delete(d.id)}
      if(d.method==="Runtime.exceptionThrown") errors.push(d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text);
    });
    const cmd=(method,params={})=>new Promise((resolve,reject)=>{
      pending.set(++seq,d=>d.error?reject(Error(d.error.message)):resolve(d.result));
      socket.send(JSON.stringify({id:seq,method,params}));
    });
    const ev=async expression=>{
      const r=await cmd("Runtime.evaluate",{expression,returnByValue:true,awaitPromise:true});
      if (r.exceptionDetails) throw Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
      return r.result.value;
    };
    const wait=async expr=>{
      for(let i=0;i<200;i++) {
        try { if(await ev(expr)) return; } catch {}
        await sleep(150);
      }
      throw Error(expr+" / "+await ev("document.body.innerText.slice(0,600)")+" / "+errors.join());
    };
    const save=(file,uri)=>fs.writeFileSync(path.join(out,file),Buffer.from(uri.split(",")[1],"base64"));
    await cmd("Page.enable");await cmd("Runtime.enable");await cmd("Network.enable");
    await cmd("Network.setCacheDisabled",{cacheDisabled:true});
    await cmd("Emulation.setDeviceMetricsOverride",{width:840,height:640,deviceScaleFactor:1,mobile:false});
    await cmd("Page.navigate",{url:base+"splats.html?sampling="+sampling});
    await wait('typeof start==="function"');
    for (const id of ids) {
      await ev("location.hash="+JSON.stringify(id));
      const settled='viewer.loaded && orderReady && !sorting && frame===0 && sortedDirection && camera().forward.every((v,i)=>Math.abs(v-sortedDirection[i])<1e-6)';
      await wait('row?.id==='+JSON.stringify(id)+' && '+settled);
      await ev('pose();viewer.jumpCameraToGoal()');
      await wait(settled+' && Math.abs(viewer.getCameraOrbit().phi-initialPose.phi)<0.0001');
      const poses=[["oblique",null]];
      if(captureRoad || await ev("Boolean(row.slab)") || ["P135","P130","M11","A119","A128"].includes(id)) poses.push(["low",[122,76]],["top",[0,0]]);
      if(await ev("Boolean(row.slabDetailCamera)")) poses.push(["socket",null]);
      for (const [kind,angle] of poses) {
        if (kind==="socket") {
          await ev('el("slabDetail").click()');
          await wait(settled+' && Math.abs(viewer.getCameraOrbit().radius-row.slabDetailCamera.radius)<1e-5 && Math.abs(viewer.getCameraTarget().x-row.slabDetailCamera.target[0])<1e-5');
        } else if (angle) {
          await ev('pose('+angle.join(",")+');viewer.jumpCameraToGoal()');
          await wait(settled+' && Math.abs(viewer.getCameraOrbit().phi-'+angle[1]*Math.PI/180+')<0.0001');
        }
        const result=await ev(`(()=>{
          const cam=camera();renderer.draw(cam,canvas.width,canvas.height);
          const orbit=viewer.getCameraOrbit(),t=viewer.getCameraTarget();
          return {img:canvas.toDataURL("image/png"),mesh:viewer.toDataURL("image/png"),
            view:{key:row.id,mode:"${kind}",width:canvas.width,height:canvas.height,count:renderer.count,blend:renderer.floatTarget?"rgba16f":"rgba8",sources:row.gaussianSources,
                  camera:{theta:orbit.theta-viewer.turntableRotation,phi:orbit.phi,radius:orbit.radius,
                    target:[t.x,t.y,t.z],fov:viewer.getFieldOfView()}}}
        })()`);
        const stem=id+"-"+kind;
        save(stem+"-gaussian.png",result.img);save(stem+"-mesh.png",result.mesh);
        if(captureRoad) {
          await wait('!el("hostControls").hidden && !el("hostRoad").checked && !el("traceUnder").checked');
          result.view.road_height_m=await ev("renderer.roadHeight");
          result.view.probe_height_m=await ev("renderer.probeHeight");
          const toggle=id=>ev('el('+JSON.stringify(id)+').click()');
          const snapshot=()=>ev('renderer.draw(camera(),canvas.width,canvas.height);canvas.toDataURL("image/png")');
          await toggle("traceUnder");save(stem+"-probe.png",await snapshot());
          await toggle("hostRoad");await wait("renderer.roadActive && renderer.asphalt");
          save(stem+"-road-probe.png",await snapshot());
          await toggle("traceUnder");save(stem+"-road.png",await snapshot());
          if (await ev("Boolean(row.pitRoad)")) {
            for (const [suffix,pass] of [["footprint","cutout"],["base","road"],["unblended","hard"]]) {
              save(stem+"-"+suffix+".png",await ev('renderer.draw(camera(),canvas.width,canvas.height,'+JSON.stringify(pass)+');canvas.toDataURL("image/png")'));
            }
          }
          await toggle("hostRoad");
          if (await ev("renderer.gl.getError()")!==0) throw Error("WebGL error during road rendering");
        }
        fs.writeFileSync(path.join(out,stem+".json"),JSON.stringify(result.view,null,2)+"\n");
      }
      await ev('el("split").click();el("divider").value=25;el("divider").dispatchEvent(new Event("input"))');
      await wait('el("split").getAttribute("aria-pressed")==="true" && !el("boundary").hidden && canvas.style.clipPath.includes("75%")');
      await ev('el("meshMode").click()');await wait('canvas.hidden && el("meshMode").getAttribute("aria-pressed")==="true"');
      console.log(id,poses.length);
    }
    if(errors.length) throw Error(errors.join("\n"));
    await cmd("Browser.close");
  } finally {
    socket?.close();
    if (chrome.exitCode===null && chrome.signalCode===null) {
      const exited=new Promise(resolve=>chrome.once("exit",resolve));
      chrome.kill();await Promise.race([exited,sleep(3000)]);
    }
    fs.rmSync(profile,{recursive:true,force:true,maxRetries:5,retryDelay:200});
  }
}
capture().catch(error=>{console.error(error);process.exitCode=1});
