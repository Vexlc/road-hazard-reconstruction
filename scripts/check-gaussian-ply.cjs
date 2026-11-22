/** Exercise the shipped browser PLY decoder and exact depth ordering on the whole local inventory. */
const fs=require("node:fs");
const path=require("node:path");
const root=path.resolve(__dirname,"../tools/asset_atlas");
const rows=JSON.parse(fs.readFileSync(path.join(root,"catalog.js"),"utf8").slice("window.roadAssets=".length,-2));
const gaussianWorker=require(path.join(root,"splat-worker.js"));
const worker={};
let response;
worker.postMessage=message=>{if(message.type==="error") throw Error(message.message);response=message;};
gaussianWorker(worker);

async function check() {
  let total=0;
  const methods={};
  const axis=[-.25,.47,-.846,2.5].map(Math.fround);
  for (const row of rows) {
    const buffer=fs.readFileSync(path.resolve(root,row.ply));
    const file={arrayBuffer:()=>buffer.buffer.slice(buffer.byteOffset,buffer.byteOffset+buffer.byteLength)};
    await worker.onmessage({data:{type:"load",file,count:row.gaussians,textureLimit:2048}});
    const {packed,count,lo,hi}=response;
    if(count!==row.gaussians || lo.some((v,i)=>!Number.isFinite(v) || !(hi[i]>v))) throw Error(row.id+" bounds or point count");
    await worker.onmessage({data:{type:"sort",forward:axis}});
    const order=response.order;
    if(order.length!==count) throw Error(row.id+" missing splats");
    let last=Infinity,previous=-1;
    for(const index of order) {
      const p=index*24;
      const depth=Math.fround(Math.fround(packed[p]*axis[0]+packed[p+1]*axis[1]+packed[p+2]*axis[2])+axis[3]);
      if(depth>last || (depth===last && index>=previous) || index>=count) throw Error(row.id+" misordered or duplicate splat");
      last=depth;previous=index;
    }
    methods[row.method]=(methods[row.method]||0)+1;
    total+=count;
  }
  const summary={decoded_assets:rows.length,sorted_points:total,methods,texture_dimension_limit:2048,depth_row:axis};
  const json=JSON.stringify(summary,null,2)+"\n";
  if(process.argv[2])fs.writeFileSync(process.argv[2],json);
  else process.stdout.write(json);
}
check().catch(error=>{console.error(error);process.exitCode=1});
