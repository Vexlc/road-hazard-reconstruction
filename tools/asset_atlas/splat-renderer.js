// Camera vectors, Gaussian means and covariances use the original PLY Z-up frame.
class GaussianRenderer {
  constructor(canvas) {
    const gl=canvas.getContext("webgl2",{alpha:false,antialias:false,depth:false,stencil:false});
    if (!gl) throw Error("此设备需要启用 WebGL 2 才能查看高斯资产。");
    this.gl=gl;
    const vertex=`#version 300 es
precision highp float;
precision highp int;
layout(location=0) in uint splatIndex;
uniform highp sampler2D gaussians;
uniform int textureWidth;
uniform vec3 eye, right, up, forward;
uniform vec2 viewport;
uniform float focal;
uniform bool roadPass, traceUnder;
out vec2 delta;
flat out vec3 conic, colour;
flat out float opacity, height;
vec4 datum(int index) {
  return texelFetch(gaussians,ivec2(index%textureWidth,index/textureWidth),0);
}
void main() {
  if (roadPass) {
    gl_Position=vec4(gl_VertexID==1?3.:-1.,gl_VertexID==2?3.:-1.,0.,1.);
    return;
  }
  int index=int(splatIndex)*6;
  vec4 center=datum(index);
  height=center.z;
  vec3 v=center.xyz-eye;
  float z=dot(v,forward);
  if (z<=0.001) { gl_Position=vec4(2.,2.,2.,1.); return; }
  vec2 uv=vec2(dot(v,right),dot(v,up))/z;
  vec3 diagonal=datum(index+1).xyz, off=datum(index+2).xyz;
  mat3 cov=mat3(diagonal.x,off.x,off.y,off.x,diagonal.y,off.z,off.y,off.z,diagonal.z);
  vec3 dx=focal/z*(right-uv.x*forward), dy=focal/z*(up-uv.y*forward);
  float a=dot(dx,cov*dx), b=dot(dx,cov*dy), c=dot(dy,cov*dy);
  float rawDet=max(0.,a*c-b*b);
  a+=0.3; c+=0.3;
  float det=a*c-b*b;
  conic=vec3(c,-b,a)/det;
  opacity=center.w*sqrt(rawDet/det);
  vec2 corner=vec2((gl_VertexID&1)==0?-1.:1.,(gl_VertexID&2)==0?-1.:1.);
  delta=corner*ceil(3.34*sqrt(vec2(a,c)));
  gl_Position=vec4(2.*(focal*uv+delta)/viewport,0.,1.);
  vec4 sh=vec4(1.,normalize(v));
  colour=max(vec3(0.),vec3(dot(sh,datum(index+3)),dot(sh,datum(index+4)),dot(sh,datum(index+5))));
  if (traceUnder && height < -0.075) colour=vec3(1.,0.04,0.6);
}`;
    const fragment=`#version 300 es
precision highp float;
in vec2 delta;
flat in vec3 conic, colour;
flat in float opacity, height;
uniform bool roadPass, roadEnabled;
uniform vec3 eye, right, up, forward;
uniform vec2 viewport;
uniform float focal, tileMetres, roadHeight;
uniform vec4 maskBounds;
uniform sampler2D roadTexture, openingMask;
out vec4 pixel;
void main() {
  if (roadPass || (roadEnabled && height<roadHeight)) {
    vec2 uv=(gl_FragCoord.xy-0.5*viewport)/focal;
    vec3 ray=forward+right*uv.x+up*uv.y;
    if (ray.z>=-0.000001) discard;
    vec2 xy=eye.xy+((roadHeight-eye.z)/ray.z)*ray.xy;
    vec2 maskUV=(xy-maskBounds.xy)/maskBounds.zw;
    bool inBox=all(greaterThanEqual(maskUV,vec2(0.))) && all(lessThanEqual(maskUV,vec2(1.)));
    bool opening=inBox && textureLod(openingMask,maskUV,0.).r>0.5;
    if (roadPass) {
      if (opening) discard;
      pixel=vec4(texture(roadTexture,xy/tileMetres).rgb,1.);
      return;
    }
    if (!opening) discard;
  }
  float power=-0.5*(conic.x*delta.x*delta.x+conic.z*delta.y*delta.y)-conic.y*delta.x*delta.y;
  float alpha=min(0.999,opacity*exp(min(0.,power)));
  if (alpha<1./255.) discard;
  pixel=vec4(colour*alpha,alpha);
}`;
    this.program=gl.createProgram();
    for (const [type,source] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]) {
      const shader=gl.createShader(type);
      gl.shaderSource(shader,source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
      gl.attachShader(this.program,shader); gl.deleteShader(shader);
    }
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program,gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(this.program));
    gl.useProgram(this.program);
    this.uniform={};
    for (const name of ["textureWidth","eye","right","up","forward","viewport","focal","roadPass","roadEnabled","traceUnder",
      "tileMetres","roadHeight","maskBounds","roadTexture","openingMask"]) this.uniform[name]=gl.getUniformLocation(this.program,name);
    gl.uniform1i(this.uniform.roadTexture,1);gl.uniform1i(this.uniform.openingMask,2);
    this.roadHeight=-0.003;
    gl.uniform1f(this.uniform.roadHeight,this.roadHeight);
    this.order=gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER,this.order);
    gl.enableVertexAttribArray(0); gl.vertexAttribIPointer(0,1,gl.UNSIGNED_INT,0,0); gl.vertexAttribDivisor(0,1);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(223/255,225/255,216/255,1);
    this.limit=gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.count=0;
    this.floatTarget=Boolean(gl.getExtension("EXT_color_buffer_float"));
    this.targetWidth=this.targetHeight=0;
  }
  upload(cloud) {
    const gl=this.gl;
    this.clear();
    this.texture=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,this.texture);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,cloud.width,cloud.height,0,gl.RGBA,gl.FLOAT,cloud.packed);
    gl.uniform1i(this.uniform.textureWidth,cloud.width);
    gl.bufferData(gl.ARRAY_BUFFER,cloud.count*4,gl.DYNAMIC_DRAW);
    if (gl.getError()!==gl.NO_ERROR) throw Error("当前设备无法分配此模型的显存。");
    this.count=cloud.count;
  }
  setOrder(order) {
    this.gl.bufferSubData(this.gl.ARRAY_BUFFER,0,order);
  }
  setRoad(profile) {
    const gl=this.gl;
    gl.deleteTexture(this.mask || null);this.mask=null;
    this.roadActive=this.traceUnder=false;
    if (!profile) return;
    // The local polygon is in units of 10 micrometres. A one-pixel border
    // keeps the clamp sampler outside the opening on all four bounding edges.
    const size=1024, xy=profile.xy.map(point=>point.map(v=>v/100000));
    const xs=xy.map(p=>p[0]), ys=xy.map(p=>p[1]);
    const left=Math.min(...xs), bottom=Math.min(...ys);
    const stepX=(Math.max(...xs)-left)/(size-2), stepY=(Math.max(...ys)-bottom)/(size-2);
    const mask=document.createElement("canvas");mask.width=mask.height=size;
    const ctx=mask.getContext("2d",{alpha:false});
    ctx.fillStyle="#fff";ctx.beginPath();
    for (const [i,[x,y]] of xy.entries()) {
      const u=(x-left)/stepX+1,v=(y-bottom)/stepY+1;
      if (i===0) ctx.moveTo(u,v); else ctx.lineTo(u,v);
    }
    ctx.closePath();ctx.fill();
    this.mask=gl.createTexture();
    gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,this.mask);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,mask);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.uniform4f(this.uniform.maskBounds,left-stepX,bottom-stepY,stepX*size,stepY*size);
    gl.activeTexture(gl.TEXTURE0);
  }
  async loadRoadTexture(material) {
    if (this.asphalt) return;
    if (!this.asphaltImage) {
      this.asphaltImage=new Image();this.asphaltImage.src=material.texture;
    }
    await this.asphaltImage.decode();
    if (this.asphalt) return;
    const gl=this.gl;
    this.asphalt=gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.asphalt);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,this.asphaltImage);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.uniform1f(this.uniform.tileMetres,material.tile_m);
    gl.activeTexture(gl.TEXTURE0);
  }
  target(width,height) {
    const gl=this.gl;
    if (!this.floatTarget) return;
    if (width!==this.targetWidth || height!==this.targetHeight) {
      gl.deleteFramebuffer(this.framebuffer || null);
      gl.deleteRenderbuffer(this.colourBuffer || null);
      this.framebuffer=gl.createFramebuffer();this.colourBuffer=gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER,this.colourBuffer);
      gl.renderbufferStorage(gl.RENDERBUFFER,gl.RGBA16F,width,height);
      gl.bindFramebuffer(gl.FRAMEBUFFER,this.framebuffer);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.RENDERBUFFER,this.colourBuffer);
      this.floatTarget=gl.checkFramebufferStatus(gl.FRAMEBUFFER)===gl.FRAMEBUFFER_COMPLETE;
      this.targetWidth=width;this.targetHeight=height;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER,this.floatTarget?this.framebuffer:null);
  }
  draw(camera,width,height) {
    const gl=this.gl;
    if (gl.canvas.width!==width || gl.canvas.height!==height) {gl.canvas.width=width;gl.canvas.height=height;}
    this.target(width,height);
    gl.viewport(0,0,width,height); gl.clear(gl.COLOR_BUFFER_BIT);
    for (const name of ["eye","right","up","forward"]) gl.uniform3fv(this.uniform[name],camera[name]);
    gl.uniform2f(this.uniform.viewport,width,height);
    gl.uniform1f(this.uniform.focal,height/(2*Math.tan(camera.fov*Math.PI/360)));
    const hosted=Boolean(this.roadActive && this.mask && this.asphalt && camera.eye[2]>this.roadHeight);
    gl.uniform1i(this.uniform.roadEnabled,hosted);
    gl.uniform1i(this.uniform.traceUnder,Boolean(this.traceUnder));
    if (hosted) {
      gl.uniform1i(this.uniform.roadPass,1);
      gl.drawArraysInstanced(gl.TRIANGLES,0,3,1);
    }
    gl.uniform1i(this.uniform.roadPass,0);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,this.count);
    if (this.floatTarget) {
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null);
      gl.blitFramebuffer(0,0,width,height,0,0,width,height,gl.COLOR_BUFFER_BIT,gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER,null);
    }
  }
  clear() {
    this.gl.deleteTexture(this.texture || null);
    this.texture=null; this.count=0;
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);
  }
}
