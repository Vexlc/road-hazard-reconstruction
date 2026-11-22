(() => {
  const byId = id => document.getElementById(id);
  const panel = byId("sourceProbe"), canvas = byId("probeMap"), mode = byId("probeView");
  const context = canvas.getContext("2d"), output = byId("probeResult");
  const cache = new Map(), loads = new Map(), images = new Map();
  let row, data, base, point, revision = 0;

  function bytes(text) {
    return Uint8Array.from(atob(text), ch => ch.charCodeAt(0));
  }
  window.sourceHeightProbe = (id, raw) => {
    const heights = text => new DataView(bytes(text).buffer);
    cache.set(id, {...raw, raw: heights(raw.raw), mesh: heights(raw.mesh),
                   flags: bytes(raw.flags), first: bytes(raw.first)});
  };
  function load(row) {
    if (cache.has(row.id)) return Promise.resolve(cache.get(row.id));
    if (!loads.has(row.id)) {
      const script = document.createElement("script");
      script.src = row.sourceProbe.data;
      loads.set(row.id, new Promise((resolve, reject) => {
        script.onload = () => cache.has(row.id) ? resolve(cache.get(row.id)) : reject(Error("编号不符"));
        script.onerror = () => reject(Error("量程文件读取失败"));
      }).finally(() => { loads.delete(row.id); script.remove(); }));
      document.head.append(script);
    }
    return loads.get(row.id);
  }
  function picture(url) {
    if (!images.has(url)) {
      const im = new Image();
      const wait = new Promise((resolve, reject) => {
        im.onload = () => resolve(im);
        im.onerror = () => { images.delete(url); reject(Error("底图读取失败")); };
      });
      images.set(url, wait);
      im.src = url;
    }
    return images.get(url);
  }
  function heightAt(array, i) {
    const v = array.getUint16(2 * i, true);
    return v ? (v - 32768) / 10 : null;
  }
  function mm(v) {
    return v === null ? "缺测" : (v > 0 ? "+" : "") + v.toFixed(1) + " mm";
  }
  function draw() {
    if (!base || !point || !data) return;
    const [x, y] = point, i = y * data.width + x;
    context.drawImage(base, 0, 0);
    context.beginPath();
    context.moveTo(x-9,y+.5); context.lineTo(x+10,y+.5);
    context.moveTo(x+.5,y-9); context.lineTo(x+.5,y+10);
    for (const [color,w] of [["#fff",3],["#132922",1]]) {
      context.strokeStyle = color; context.lineWidth = w; context.stroke();
    }
    const prefix = "像素 (" + x + ", " + y + ")　";
    const f = data.flags[i];
    if (!(f & 1)) { output.textContent = prefix + "目标范围外"; return; }
    const a = heightAt(data.raw, i), b = heightAt(data.mesh, i);
    const region = ["坑内区域", "砖顶面", "砖侧面"][(f >> 2) & 3];
    const surface = {0:"无交点",2:"坑面",13:"砖顶面",14:"砖侧面"}[data.first[i]] || "相邻路面";
    const parts = [region, f & 2 ? "原坑洞标注内" : "补入区域", "实测 " + mm(a), "网格 " + mm(b)];
    if (a !== null && b !== null) parts.push("网格减实测 " + mm(b-a));
    parts.push(a === null ? "原始传感器无量程" : (f & 16 ? "预留验证像素" : "拟合像素"), "网格首个交点为" + surface);
    output.textContent = prefix + parts.join("　·　");
  }
  async function show() {
    const current = ++revision;
    if (!panel.open || !row?.sourceProbe) return;
    base = null;
    context.clearRect(0,0,canvas.width,canvas.height);
    output.textContent = "正在读取实拍数据…";
    try {
      const [next, img] = await Promise.all([load(row), picture(row.sourceProbe[mode.value])]);
      if (current !== revision) return;
      if (img.width !== next.width || img.height !== next.height) throw Error("量程与底图尺寸不一致");
      data = next; base = img;
      canvas.width = data.width; canvas.height = data.height;
      if (!point) point = [...data.examples[0].xy];
      byId("probeExamples").replaceChildren(...data.examples.map(example => {
        const button = document.createElement("button");
        button.type = "button"; button.textContent = example.label;
        button.onclick = () => { point = [...example.xy]; draw(); };
        return button;
      }));
      draw();
    } catch (e) { if (current === revision) output.textContent = e.message; }
  }
  window.showSourceProbe = selected => {
    revision++; row = selected; point = data = base = null;
    panel.hidden = !row.sourceProbe; panel.open = false; mode.value = "rgb";
    byId("probeExamples").replaceChildren();
  };
  panel.ontoggle = show; mode.onchange = show;
  canvas.onclick = event => {
    if (!data || !base) return;
    const r = canvas.getBoundingClientRect();
    point = [Math.min(data.width-1, Math.floor((event.clientX-r.left)*data.width/r.width)),
             Math.min(data.height-1, Math.floor((event.clientY-r.top)*data.height/r.height))];
    draw(); canvas.focus();
  };
  canvas.onkeydown = event => {
    const d = {ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]}[event.key];
    if (!d || !point || !base) return;
    event.preventDefault();
    point = [Math.min(data.width-1,Math.max(0,point[0]+d[0])),Math.min(data.height-1,Math.max(0,point[1]+d[1]))];
    draw();
  };
})();
