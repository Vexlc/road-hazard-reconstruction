// Pack the original PLY once. Sort camera-space depths without blocking pointer input.
function gaussianWorker(scope) {
  "use strict";
  scope=scope || self;
  let xyz, keys, indices, bins;
  function unpack(buffer, textureLimit, expectedCount) {
    const bytes=new Uint8Array(buffer);
    const prefix=new TextDecoder("ascii").decode(bytes.subarray(0,Math.min(65536,bytes.length)));
    const end=/^end_header\r?\n/m.exec(prefix);
    if (!end) throw Error("PLY 缺少文件头。");
    const offset=end.index+end[0].length;
    const lines=prefix.slice(0,end.index).trim().split(/\r?\n/);
    if (lines[0]!=="ply" || lines[1]!=="format binary_little_endian 1.0") throw Error("需要小端二进制 3DGS PLY。");
    const fields=[];
    let count=0, element="";
    for (const line of lines.slice(2)) {
      const [op,type,name]=line.trim().split(/\s+/);
      if (op==="element") {
        element=type;
        if (element==="vertex") count=Number(name);
        else if (Number(name)>0) throw Error("需要仅含高斯顶点的 PLY。");
      } else if (op==="property" && element==="vertex") {
        if (!["float","float32"].includes(type) || fields.includes(name)) throw Error("不支持的 PLY 顶点属性。");
        fields.push(name);
      }
    }
    const stride=fields.length;
    if (!Number.isSafeInteger(count) || count<=0 || offset+count*stride*4!==bytes.length) throw Error("PLY 点数与数据长度不一致。");
    if (expectedCount && expectedCount!==count) throw Error("PLY 点数与所选资产不一致。");
    const width=Math.min(4096,textureLimit,count*6), height=Math.ceil(count*6/width);
    if (height>textureLimit) throw Error("模型超过当前设备的纹理容量。");
    const field=name=>{
      const index=fields.indexOf(name);
      if (index<0) throw Error("PLY 缺少 "+name);
      return index;
    };
    const pos=["x","y","z"].map(field);
    const props=(stem,size)=>Array.from({length:size},(_,i)=>field(stem+i));
    const rot=props("rot_",4), scale=props("scale_",3), dc=props("f_dc_",3), rest=props("f_rest_",9);
    if (fields.includes("f_rest_9")) throw Error("当前查看器支持一阶球谐 PLY。");
    const opacity=field("opacity");
    // A PLY header can be unaligned. Reuse its buffer for aligned float32 reads.
    bytes.copyWithin(0,offset);
    const raw=new Float32Array(buffer,0,count*stride);
    const packed=new Float32Array(width*height*4);
    xyz=new Float32Array(count*3);
    const lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
    for (let i=0;i<count;i++) {
      const s=i*stride, t=i*24;
      for (let k=0;k<3;k++) {
        const value=raw[s+pos[k]];
        xyz[3*i+k]=packed[t+k]=value;
        lo[k]=Math.min(lo[k],value); hi[k]=Math.max(hi[k],value);
      }
      packed[t+3]=1/(1+Math.exp(-raw[s+opacity]));
      let w=raw[s+rot[0]], x=raw[s+rot[1]], y=raw[s+rot[2]], z=raw[s+rot[3]];
      const inverse=1/Math.sqrt(w*w+x*x+y*y+z*z);
      w*=inverse; x*=inverse; y*=inverse; z*=inverse;
      const a=1-2*(y*y+z*z), b=2*(x*y-z*w), c=2*(x*z+y*w);
      const d=2*(x*y+z*w), e=1-2*(x*x+z*z), f=2*(y*z-x*w);
      const g=2*(x*z-y*w), h=2*(y*z+x*w), j=1-2*(x*x+y*y);
      const vx=Math.exp(2*raw[s+scale[0]]), vy=Math.exp(2*raw[s+scale[1]]), vz=Math.exp(2*raw[s+scale[2]]);
      packed[t+4]=a*a*vx+b*b*vy+c*c*vz;
      packed[t+5]=d*d*vx+e*e*vy+f*f*vz;
      packed[t+6]=g*g*vx+h*h*vy+j*j*vz;
      packed[t+8]=a*d*vx+b*e*vy+c*f*vz;
      packed[t+9]=a*g*vx+b*h*vy+c*j*vz;
      packed[t+10]=d*g*vx+e*h*vy+f*j*vz;
      // Each colour channel has its own DC term and camera-to-point SH direction.
      for (let k=0;k<3;k++) {
        const base=t+12+4*k, coeff=3*k;
        packed[base]=0.5+0.28209479177387814*raw[s+dc[k]];
        packed[base+1]=-0.4886025119029199*raw[s+rest[coeff+2]];
        packed[base+2]=-0.4886025119029199*raw[s+rest[coeff]];
        packed[base+3]=0.4886025119029199*raw[s+rest[coeff+1]];
      }
    }
    for (let i=0;i<count*24;i++) if (!Number.isFinite(packed[i])) throw Error("PLY 含无效的坐标、旋转或球谐参数。");
    keys=new Uint32Array(count); indices=new Uint32Array(count); bins=new Uint32Array(65536);
    return {packed,width,height,count,lo,hi};
  }

  function sort(forward, recycle) {
    const count=keys.length;
    const depths=new Float32Array(keys.buffer);
    const result=recycle?.byteLength===count*4 ? new Uint32Array(recycle) : new Uint32Array(count);
    bins.fill(0);
    for (let i=0;i<count;i++) {
      depths[i]=Math.fround(xyz[i*3]*forward[0]+xyz[i*3+1]*forward[1]+xyz[i*3+2]*forward[2])+forward[3];
      const bits=keys[i];
      // The ascending unsigned key orders signed float depths from far to near.
      keys[i]=bits & 0x80000000 ? bits : (bits ^ 0x7fffffff);
      bins[keys[i]&65535]++;
    }
    function prefix() {
      let start=0;
      for (let b=0;b<bins.length;b++) { const n=bins[b];bins[b]=start;start+=n; }
    }
    prefix();
    // CUDA composites equal-depth points front-to-back in PLY order. Our over blend traverses them backwards.
    for (let i=count-1;i>=0;i--) indices[bins[keys[i]&65535]++]=i;
    bins.fill(0);
    for (let i=0;i<count;i++) bins[keys[i]>>>16]++;
    prefix();
    for (let i=0;i<count;i++) { const index=indices[i]; result[bins[keys[index]>>>16]++]=index; }
    return result;
  }
  scope.onmessage=async({data})=>{
    try {
      if (data.type==="load") {
        let buffer;
        if (data.file) buffer=await data.file.arrayBuffer();
        else {
          const response=await fetch(data.url);
          if (!response.ok) throw Error("无法读取资产 PLY（HTTP "+response.status+"）。");
          buffer=await response.arrayBuffer();
        }
        const cloud=unpack(buffer,data.textureLimit,data.count);
        scope.postMessage({type:"cloud",...cloud},[cloud.packed.buffer]);
      } else if (data.type==="sort") {
        const order=sort(data.forward,data.recycle);
        scope.postMessage({type:"order",order,forward:data.forward},[order.buffer]);
      }
    } catch (err) { scope.postMessage({type:"error",message:err.message}); }
  };
}

if (typeof module!=="undefined" && module.exports) module.exports=gaussianWorker;
