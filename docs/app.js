/* Kisan from Space: map app. Plain JavaScript + Leaflet, no build step. */
"use strict";

const NOT_CROP = 255;
const INDIA = [[6, 68], [37, 98]];
const state = { meta: null, areas: [], classes: [], selected: null, filter: null, minConf: 0,
                layerMode: "crops", opacity: 0.8, coverage: null, tiles: [], fieldsLayer: null };
const TILE_ZOOM = 11, FIELD_ZOOM = 13;   // show training tiles from zoom 11, farm outlines from zoom 13

// ---------------- map + base layers ----------------
const map = L.map("map", { zoomControl: false, preferCanvas: true, worldCopyJump: true }).fitBounds(INDIA);
L.control.zoom({ position: "topright" }).addTo(map);
L.control.scale({ position: "topright", imperial: false }).addTo(map);

const base = {
  sat: L.layerGroup([
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, maxNativeZoom: 18, attribution: "Imagery © Esri, Maxar, Earthstar Geographics" }),
    L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, maxNativeZoom: 18, attribution: "Labels © Esri" }),
  ]),
  map: L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 19, attribution: "© OpenStreetMap contributors" }),
};
base.sat.addTo(map);

document.querySelectorAll("[data-base]").forEach(b => b.addEventListener("click", () => {
  Object.values(base).forEach(l => map.removeLayer(l));
  base[b.dataset.base].addTo(map);
  base[b.dataset.base].bringToBack?.();
  document.querySelectorAll("[data-base]").forEach(x => x.setAttribute("aria-pressed", x === b));
}));

// ---------------- helpers ----------------
const $ = s => document.querySelector(s);
const fmt = (v, d = 0) => Number(v).toLocaleString("en-IN", { maximumFractionDigits: d, minimumFractionDigits: d });
const hexToRgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
function km(a, b) {  // great-circle distance in km between [lat, lon] pairs
  const R = 6371, r = x => x * Math.PI / 180;
  const dLat = r(b[0] - a[0]), dLon = r(b[1] - a[1]);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(r(a[0])) * Math.cos(r(b[0])) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
let toastTimer;
function toast(html, ms = 7000) {
  const t = $("#toast"); t.innerHTML = html; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), ms);
}
const colourOf = name => state.classes.find(c => c.name === name)?.colour || "#999";

// ---------------- crop layers (drawn in the browser from grid.bin) ----------------
function renderArea(a) {
  const { width: w, height: h } = a.grid;
  const cv = a.canvas || (a.canvas = Object.assign(document.createElement("canvas"), { width: w, height: h }));
  const ctx = cv.getContext("2d"), img = ctx.createImageData(w, h), d = img.data, g = a.data;
  const rgb = state.classes.map(c => hexToRgb(c.colour));
  for (let i = 0, p = 0; i < w * h; i++, p += 2) {
    const cls = g[p], conf = g[p + 1], o = i * 4;
    if (cls === NOT_CROP || conf < state.minConf) { d[o + 3] = 0; continue; }
    const [r, gg, b] = rgb[cls];
    const dim = state.filter !== null && cls !== state.filter;
    d[o] = r; d[o + 1] = gg; d[o + 2] = b; d[o + 3] = dim ? 40 : 255;
  }
  ctx.putImageData(img, 0, 0);
  const url = cv.toDataURL();
  if (a.cropLayer) a.cropLayer.setUrl(url);
  else a.cropLayer = L.imageOverlay(url, a.bounds, { className: "crop-layer", opacity: state.opacity, interactive: false });
}
function applyLayerMode() {
  for (const a of state.areas) {
    [a.cropLayer, a.s2Layer].forEach(l => l && map.removeLayer(l));
    if (state.layerMode === "crops") a.cropLayer.setOpacity(state.opacity).addTo(map);
    if (state.layerMode === "s2") a.s2Layer.setOpacity(Math.max(state.opacity, 0.9)).addTo(map);
  }
  updateTiles();
}
document.querySelectorAll("[data-layer]").forEach(b => b.addEventListener("click", () => {
  state.layerMode = b.dataset.layer;
  document.querySelectorAll("[data-layer]").forEach(x => x.setAttribute("aria-pressed", x === b));
  applyLayerMode();
}));
$("#opacity").addEventListener("input", e => { state.opacity = +e.target.value; applyLayerMode(); });
$("#minconf").addEventListener("input", e => {
  state.minConf = +e.target.value; $("#conf-val").textContent = state.minConf;
  state.areas.forEach(renderArea);
});

// ---------------- all 1,165 training tiles + surveyed farms ----------------
const tileLayers = new Map();   // chip -> { crop: overlay, img: overlay }, created only when first needed
function updateTiles() {
  const zoom = map.getZoom(), on = $("#show-coverage").checked;
  const show = zoom >= TILE_ZOOM && state.layerMode !== "off";
  const view = map.getBounds().pad(0.2);
  for (const t of state.tiles) {
    const visible = show && view.intersects(t.b);
    let l = tileLayers.get(t.chip);
    if (visible && !l) {
      l = { crop: L.imageOverlay(`data/tiles/${t.chip}.png`, t.b, { className: "crop-layer" }),
            img: L.imageOverlay(`data/tiles/${t.chip}.jpg`, t.b, { className: "crop-layer" }) };
      tileLayers.set(t.chip, l);
    }
    if (!l) continue;
    const want = visible ? (state.layerMode === "crops" ? l.crop : l.img) : null;
    for (const layer of [l.crop, l.img]) {
      if (layer === want) { layer.setOpacity(layer === l.crop ? state.opacity : 1); if (!map.hasLayer(layer)) layer.addTo(map); }
      else if (map.hasLayer(layer)) map.removeLayer(layer);
    }
  }
  if (state.coverage) (on && zoom < TILE_ZOOM) ? state.coverage.addTo(map) : map.removeLayer(state.coverage);
  if (state.fieldsLayer) (on && zoom >= FIELD_ZOOM) ? state.fieldsLayer.addTo(map) : map.removeLayer(state.fieldsLayer);
}
map.on("zoomend moveend", updateTiles);

const tileAt = latlng => state.tiles.find(t => L.latLngBounds(t.b).contains(latlng));
let colourIndex;
async function tilePixel(t, latlng) {
  if (!t.px) {   // read the tile's PNG once and keep its pixels
    const img = new Image(); img.src = `data/tiles/${t.chip}.png`; await img.decode();
    const c = Object.assign(document.createElement("canvas"), { width: img.width, height: img.height });
    const ctx = c.getContext("2d"); ctx.drawImage(img, 0, 0);
    t.px = ctx.getImageData(0, 0, c.width, c.height).data;
  }
  colourIndex ||= new Map(state.classes.map((c, i) => [hexToRgb(c.colour).join(","), i]));
  const [x0, y0, dx, dy, W, H] = t.g, p = L.CRS.EPSG3857.project(latlng);
  const col = Math.floor((p.x - x0) / dx), row = Math.floor((p.y - y0) / dy);
  if (col < 0 || row < 0 || col >= W || row >= H) return null;
  const o = (row * W + col) * 4;
  return t.px[o + 3] === 0 ? NOT_CROP : colourIndex.get(`${t.px[o]},${t.px[o + 1]},${t.px[o + 2]}`) ?? null;
}
function fieldPopup(p) {
  const role = { train: "used to train the AI", val: "used to tune the AI", test: "test farm: the AI never saw it" }[p.split];
  const ai = p.ai ? `<br>AI said: <b>${p.ai}</b> (${Math.round(p.conf * 100)}% sure) · ${p.ai === p.crop ? "✓ right" : "✗ wrong"}` : "";
  return `<div class="pop"><b>Surveyed farm · ${p.ha} ha</b><br>Farmer said: <b>${p.crop}</b>${ai}<br><span class="m">${role}</span></div>`;
}
function drawFields(fc) {
  state.fieldsLayer = L.geoJSON(fc, {
    // filled with the FARMER's crop colour; white outline = AI right, red dashed outline = AI wrong
    style: f => {
      const wrong = f.properties.ai && f.properties.ai !== f.properties.crop;
      return { fillColor: colourOf(f.properties.crop), fillOpacity: 0.9, color: wrong ? "#FF3B30" : "#FFFFFF",
               weight: 2, opacity: 1, dashArray: wrong ? "4 3" : null };
    },
    bubblingMouseEvents: false,
    onEachFeature: (f, layer) => layer.bindPopup(fieldPopup(f.properties)),
  });
  updateTiles();
}

// ---------------- click a pixel ----------------
function pixelAt(a, latlng) {
  const p = L.CRS.EPSG3857.project(latlng);
  const col = Math.floor((p.x - a.grid.x0) / a.grid.dx), row = Math.floor((p.y - a.grid.y0) / a.grid.dy);
  if (col < 0 || row < 0 || col >= a.grid.width || row >= a.grid.height) return null;
  const k = (row * a.grid.width + col) * 2;
  return { cls: a.data[k], conf: a.data[k + 1] };
}
const areaAt = latlng => state.areas.find(a => L.latLngBounds(a.bounds).contains(latlng));
map.on("click", async e => {
  const a = areaAt(e.latlng), coords = `${e.latlng.lat.toFixed(5)}° N, ${e.latlng.lng.toFixed(5)}° E`;
  let html;
  const t = !a && tileAt(e.latlng);
  if (t) {
    const cls = await tilePixel(t, e.latlng);
    html = cls === NOT_CROP || cls === null
      ? `<div class="pop"><b>Not cropland</b><br>Village, road, water or trees (ESA WorldCover).<br><span class="m">${coords}</span></div>`
      : `<div class="pop"><b style="color:${state.classes[cls].colour};filter:brightness(.8)">■</b> <b>${state.classes[cls].name}</b><br>
         AI prediction · training tile ${t.chip}, spring 2022<br><span class="m">${coords}</span></div>`;
  } else if (!a) html = `<div class="pop">No AI crop map here.<br><span class="m">${coords}</span></div>`;
  else {
    const px = pixelAt(a, e.latlng);
    if (!px || px.cls === NOT_CROP)
      html = `<div class="pop"><b>Not cropland</b><br>Village, road, water or trees (ESA WorldCover).<br><span class="m">${coords}</span></div>`;
    else {
      const c = state.classes[px.cls];
      html = `<div class="pop"><b style="color:${c.colour};filter:brightness(.8)">■</b> <b>${c.name}</b><br>
        AI prediction, ${px.conf}% sure · image of ${a.date}<br><span class="m">${coords}</span></div>`;
    }
    if (state.selected !== a) selectArea(a, false);
  }
  L.popup().setLatLng(e.latlng).setContent(html).openOn(map);
});

// ---------------- sidebar: areas ----------------
function barHtml(crops) {
  const total = crops.reduce((s, c) => s + c.ha, 0) || 1;
  return `<div class="bar">${crops.map(c => `<span title="${c.crop}" style="width:${c.ha / total * 100}%;background:${colourOf(c.crop)}"></span>`).join("")}</div>`;
}
function listAreas() {
  $("#area-list").innerHTML = state.areas.map((a, i) => `<li><button data-i="${i}" aria-current="${state.selected === a}">
      <span class="t">${a.title}</span><span class="d">image ${a.date} · ${a.size_km} × ${a.size_km} km</span>${barHtml(a.crops)}</button></li>`).join("");
  $("#area-list").querySelectorAll("button").forEach(b => b.addEventListener("click", () => selectArea(state.areas[+b.dataset.i], true)));
}
function selectArea(a, fly) {
  state.selected = a; state.filter = null;
  const shown = a.crops.filter(c => c.ha >= 1);          // hide crops with less than 1 ha
  state.areas.forEach(renderArea);
  if (fly) map.flyToBounds(a.bounds, { padding: [30, 30], duration: 1.2 });
  const total = a.crops.reduce((s, c) => s + c.ha, 0);
  const month = new Date(a.date).toLocaleString("en-IN", { month: "long" });
  $("#area-detail").innerHTML = `<div class="detail">
      <h2>${a.title}</h2>
      <div class="facts">
        <div><b>${fmt(total)}</b><span>ha of cropland</span></div>
        <div><b>${a.date.slice(8)} ${month.slice(0, 3)}</b><span>image date, ${a.date.slice(0, 4)}</span></div>
        <div><b>${a.mean_confidence ?? "–"}%</b><span>average AI confidence</span></div>
      </div>
      <h3 class="label">AI crop estimate · tap a crop to highlight it</h3>
      <ul class="crops">${shown.map(c => `<li><button data-crop="${c.crop}" aria-pressed="false">
          <i style="background:${colourOf(c.crop)}"></i>
          <span class="n">${c.crop}<em style="width:${Math.max(c.ha / total * 100, 1)}%;background:${colourOf(c.crop)}"></em></span>
          <span class="v">${fmt(c.ha)} ha · ${fmt(c.ha / total * 100)}%</span></button></li>`).join("")}</ul>
      <p class="note">This photo was taken in ${month}. ${+a.date.slice(5, 7) >= 4
        ? "By then many winter crops (especially mustard) were already harvested, so they are easy to miss."
        : "Crops were still standing, which gives the AI its best chance."}</p>
    </div>`;
  $("#area-detail").querySelectorAll("[data-crop]").forEach(b => b.addEventListener("click", () => {
    const idx = state.classes.findIndex(c => c.name === b.dataset.crop);
    state.filter = state.filter === idx ? null : idx;
    $("#area-detail").querySelectorAll("[data-crop]").forEach(x => x.setAttribute("aria-pressed", state.filter !== null && x === b));
    state.areas.forEach(renderArea);
  }));
  listAreas();
  showTab("areas");
}

// ---------------- training coverage dots ----------------
function drawCoverage(points) {
  state.coveragePts = points;
  state.coverage = L.layerGroup(points.map(([lat, lon, st, n]) =>
    L.circleMarker([lat, lon], { radius: 4, color: "#0b3d24", weight: 1, fillColor: "#6FCF97", fillOpacity: 0.9, bubblingMouseEvents: false })
      .bindTooltip(`Training tile · ${n} surveyed farm${n === 1 ? "" : "s"} · ${st}<br>Click to zoom in`)
      .on("click", () => map.flyTo([lat, lon], 14, { duration: 1.2 }))));
  updateTiles();
}
$("#show-coverage").addEventListener("change", updateTiles);

// ---------------- search (Photon / OpenStreetMap) ----------------
const input = $("#search"), list = $("#suggest");
let results = [], active = -1, debounce, lastQuery = "";
function parseCoords(q) {
  const m = q.match(/^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$/);
  return m ? { lat: +m[1], lon: +m[2] } : null;
}
async function suggest(q) {
  lastQuery = q;
  const c = parseCoords(q);
  if (c) { results = [{ name: `${c.lat}, ${c.lon}`, sub: "Coordinates", lat: c.lat, lon: c.lon }]; return showSuggest(); }
  if (q.length < 3) { list.hidden = true; return; }
  try {
    const r = await fetch(`https://photon.komoot.io/api/?q=${encodeURIComponent(q)}&limit=6&lang=en&bbox=68,6,98,37`);
    const j = await r.json();
    if (q !== lastQuery) return;                      // a newer search has started
    results = j.features.map(f => {
      const p = f.properties;
      return { name: p.name || p.city || p.county || q,
               sub: [p.county || p.district, p.state, p.country].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i).join(", "),
               lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], extent: p.extent };
    });
    showSuggest();
  } catch { toast("Search is unavailable right now. You can still type coordinates, e.g. 27.86, 81.35."); }
}
function showSuggest() {
  active = -1;
  if (!results.length) { list.innerHTML = `<li aria-disabled="true"><small>No places found</small></li>`; list.hidden = false; return; }
  list.innerHTML = results.map((r, i) => `<li role="option" data-i="${i}">${r.name}<small>${r.sub}</small></li>`).join("");
  list.hidden = false;
  list.querySelectorAll("[data-i]").forEach(li => li.addEventListener("mousedown", e => { e.preventDefault(); go(results[+li.dataset.i]); }));
}
function go(r) {
  list.hidden = true; input.value = r.name; input.blur();
  if (r.extent) map.flyToBounds([[r.extent[3], r.extent[0]], [r.extent[1], r.extent[2]]], { maxZoom: 15, duration: 1.4 });
  else map.flyTo([r.lat, r.lon], 14, { duration: 1.4 });
  const here = L.latLng(r.lat, r.lon), inside = areaAt(here);
  if (inside) { selectArea(inside, false); toast(`<b>${r.name}</b> has an AI crop map. Tap any field to see the predicted crop.`); return; }
  if (tileAt(here)) { toast(`<b>${r.name}</b> is inside a training tile. Zoom in to see the AI's crops and the surveyed farms.`); return; }
  const nearestArea = state.areas.map(a => ({ a, d: km([r.lat, r.lon], [a.lat, a.lon]) })).sort((x, y) => x.d - y.d)[0];
  const nearestTile = Math.min(...state.coveragePts.map(p => km([r.lat, r.lon], [p[0], p[1]])));
  const reach = nearestTile <= 50
    ? `It is ${fmt(nearestTile)} km from our training fields, so the model <b>can</b> map it: run steps 9–10 for these coordinates.`
    : `It is ${fmt(nearestTile)} km from the nearest training field, too far for this model to be trusted.`;
  toast(`<b>No AI crop map for ${r.name} yet.</b> ${reach} Nearest mapped area: ${nearestArea.a.title} (${fmt(nearestArea.d)} km).`, 10000);
}
input.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => suggest(input.value.trim()), 300); });
input.addEventListener("keydown", e => {
  const items = [...list.querySelectorAll("[data-i]")];
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault(); if (!items.length) return;
    active = (active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
    items.forEach((li, i) => li.setAttribute("aria-selected", i === active));
  } else if (e.key === "Enter") {
    e.preventDefault();
    if (results.length) go(results[Math.max(active, 0)]);
  } else if (e.key === "Escape") list.hidden = true;
});
input.addEventListener("blur", () => setTimeout(() => (list.hidden = true), 150));

// ---------------- tabs ----------------
function showTab(t) {
  document.querySelectorAll("[data-tab]").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === t));
  ["areas", "game", "about"].forEach(k => ($(`#pane-${k}`).hidden = k !== t));
}
document.querySelectorAll("[data-tab]").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------------- accuracy tab ----------------
function drawMetrics(m) {
  if (!m) { $("#metrics").innerHTML = `<p class="fine">Run step 8 (evaluation) and re-export to show accuracy here.</p>`; return; }
  const rows = m.models.map(r => `<tr><td>${r.model}</td><td class="num">${fmt(r.accuracy * 100)}%</td><td class="num">${r.macro_f1.toFixed(2)}</td></tr>`).join("");
  const crops = m.unet_per_crop.filter(c => c.test_fields > 0).sort((a, b) => b.test_fields - a.test_fields);
  $("#metrics").innerHTML = `<h3 class="label">How well does it work? (${fmt(m.test_fields)} unseen test fields)</h3>
    <table><thead><tr><th>Model</th><th class="num">Fields right</th><th class="num">Macro F1</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="fine">Macro F1 gives every crop equal weight (0 = useless, 1 = perfect). Always guessing wheat would score 0.04.</p>
    <h3 class="label">U-Net score per crop (F1)</h3>
    ${crops.map(c => `<div class="f1"><span>${c.crop}</span><div><span style="width:${c.f1 * 100}%;background:${colourOf(c.crop)}"></span></div><b>${c.f1.toFixed(2)}</b></div>`).join("")}`;
}

// ---------------- game ----------------
const game = { items: [], you: 0, ai: 0, n: 0, cur: null };
function nextField() {
  if (!game.items.length) return;
  const it = game.items[Math.floor(Math.random() * game.items.length)];
  game.cur = it;
  $("#game-img").src = `data/game/${it.id}.png`;
  $("#game-meta").textContent = `${it.state} · ${it.ha} ha field · picture is ${it.view_m} m × ${it.view_m} m`;
  const pool = [...new Set(game.items.map(g => g.crop))].filter(c => c !== it.crop).sort(() => Math.random() - 0.5);
  const opts = [it.crop, ...pool.slice(0, 3)].sort(() => Math.random() - 0.5);
  $("#choices").innerHTML = opts.map(o => `<button data-c="${o}"><i style="background:${colourOf(o)}"></i>${o}</button>`).join("");
  $("#choices").querySelectorAll("button").forEach(b => b.addEventListener("click", () => answer(b)));
  $("#result").textContent = "";
}
function answer(btn) {
  const it = game.cur; if (!it || btn.disabled) return;
  const ok = btn.dataset.c === it.crop, aiOk = it.ai === it.crop;
  game.n++; if (ok) game.you++; if (aiOk) game.ai++;
  $("#choices").querySelectorAll("button").forEach(b => { b.disabled = true; if (b.dataset.c === it.crop) b.classList.add("right"); });
  if (!ok) btn.classList.add("wrong");
  $("#s-you").textContent = `${game.you} / ${game.n}`; $("#s-ai").textContent = `${game.ai} / ${game.n}`;
  $("#result").innerHTML = `${ok ? "Correct." : "Not quite."} It is <b>${it.crop}</b> (farmer survey). The AI said <b>${it.ai}</b> (${Math.round(it.ai_conf * 100)}% sure)${aiOk ? ", right." : ", wrong."}`;
}
$("#next").addEventListener("click", nextField);

// ---------------- load data ----------------
(async function init() {
  const get = p => fetch(p).then(r => { if (!r.ok) throw new Error(p); return r; });
  const meta = await (await get("data/areas.json")).json();
  state.classes = meta.classes;
  state.areas = await Promise.all(meta.areas.map(async a => {
    a.data = new Uint8Array(await (await get(`data/${a.name}/grid.bin`)).arrayBuffer());
    a.s2Layer = L.imageOverlay(`data/${a.name}/s2.png`, a.bounds, { className: "crop-layer" });
    L.rectangle(a.bounds, { color: "#F4E04D", weight: 1.5, fill: false, dashArray: "4 4", interactive: false }).addTo(map);
    L.marker([a.bounds[1][0], (a.bounds[0][1] + a.bounds[1][1]) / 2], { opacity: 0, interactive: false })
      .bindTooltip(a.title.split(" · ")[0], { permanent: true, direction: "top", className: "area-label" }).addTo(map);
    return a;
  }));
  state.areas.forEach(renderArea);
  applyLayerMode();
  listAreas();
  $("#area-detail").innerHTML = `<p class="lead">Search any place in India, pick a mapped area below, or click a green dot. Each dot is a 2.5 km training tile: zoom in to see the AI's crop map and the outlines of real surveyed farms.</p>`;
  get("data/coverage.json").then(r => r.json()).then(drawCoverage);
  get("data/tiles.json").then(r => r.json()).then(t => { state.tiles = t; updateTiles(); }).catch(() => {});
  get("data/fields.geojson").then(r => r.json()).then(drawFields).catch(() => {});
  get("data/metrics.json").then(r => r.json()).then(drawMetrics).catch(() => drawMetrics(null));
  get("data/game.json").then(r => r.json()).then(g => { game.items = g; nextField(); });
})().catch(err => toast(`Could not load the map data (${err.message}). Run step 12, then serve the docs folder (see README).`, 20000));