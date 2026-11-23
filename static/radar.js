// Lightweight HTML5 Canvas 2D Radar Engine (Retina + smooth interpolation)

const LEVEL_HEX = ["#34d399", "#f59e0b", "#ef4444"];
const LEVEL_SECTOR_FILL = [
  "rgba(52, 211, 153, 0.04)",
  "rgba(245, 158, 11, 0.14)",
  "rgba(239, 68, 68, 0.20)",
];

const BG = "#07090e";
const RING = "rgba(238, 240, 245, 0.09)";
const AXIS_OUTER = "rgba(238, 240, 245, 0.20)";
const AXIS_INNER = "rgba(238, 240, 245, 0.10)";
const MUTED = "#8b95a8";
const TEXT = "#eef0f5";
const FONT_MONO = '500 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
const FONT_UI = '500 11px system-ui, -apple-system, sans-serif';

export class RadarCanvas {
  constructor(canvas, tooltipEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tooltip = tooltipEl;
    this.state = null;
    this.interp = new Map();
    this.hitBoxes = [];
    this.selectedId = null;
    this._bindEvents();
    this._startLoop();
  }

  _bindEvents() {
    const pick = (clientX, clientY, toggle) => {
      const rect = this.canvas.getBoundingClientRect();
      const px = clientX - rect.left;
      const py = clientY - rect.top;
      let hit = null;
      let bestD = 22;
      for (const h of this.hitBoxes) {
        const d = Math.hypot(px - h.px, py - h.py);
        if (d < bestD) {
          bestD = d;
          hit = h;
        }
      }
      if (toggle) {
        this.selectedId = hit ? (this.selectedId === hit.track.id ? null : hit.track.id) : null;
      }
      this._renderTooltip(hit ? { px, py, track: hit.track } : null);
    };

    this.canvas.addEventListener("mousemove", (e) => pick(e.clientX, e.clientY, false));
    this.canvas.addEventListener("mouseleave", () => {
      if (this.selectedId == null) this._renderTooltip(null);
    });
    this.canvas.addEventListener("click", (e) => pick(e.clientX, e.clientY, true));
  }

  _renderTooltip(info) {
    if (!this.tooltip) return;
    if (!info) {
      this.tooltip.classList.add("hidden");
      return;
    }
    const t = info.track;
    const ttcTxt = t.ttc != null ? ` · TTC ${t.ttc.toFixed(1)}s` : "";
    const spdTxt = t.closing_speed > 0.1 ? ` · ${t.closing_speed.toFixed(1)} m/s` : "";
    this.tooltip.innerHTML =
      `<div><b>${t.name}</b> <span class="mono" style="color:var(--accent)">${t.r.toFixed(2)} m</span></div>` +
      `<div class="mono" style="color:var(--muted);font-size:11px">${t.mode || "ok"}${t.dep ? "+d" : ""}${spdTxt}${ttcTxt}</div>`;
    this.tooltip.style.left = `${Math.min(Math.max(8, info.px + 10), Math.max(8, this.canvas.clientWidth - 180))}px`;
    this.tooltip.style.top = `${Math.max(8, info.py - 48)}px`;
    this.tooltip.classList.remove("hidden");
  }

  setState(state) {
    this.state = state;
    const seen = new Set();
    for (const tr of state.tracks || []) {
      seen.add(tr.id);
      const cur = this.interp.get(tr.id);
      if (cur) {
        cur.tx = tr.X;
        cur.tz = tr.Z;
        cur.meta = tr;
      } else {
        this.interp.set(tr.id, { x: tr.X, z: tr.Z, tx: tr.X, tz: tr.Z, meta: tr });
      }
    }
    for (const id of this.interp.keys()) {
      if (!seen.has(id)) this.interp.delete(id);
    }
  }

  _startLoop() {
    const tick = () => {
      this.draw();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  _syncSize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w > 0 && h > 0 && (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr))) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  draw() {
    const { w, h } = this._syncSize();
    if (!w || !h) return;
    const ctx = this.ctx;

    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);

    const s = this.state;
    const range = s?.radar_range || 5.0;
    const hfov = s?.hfov || 70.0;
    const blind = s?.blind || 0.0;
    const ox = w / 2;
    const oy = h - 20;
    const R = Math.min(w * 0.45, h - 38);
    const scale = R / Math.max(range, 1.0);
    const halfRad = (hfov * Math.PI) / 360.0;
    const a0 = -Math.PI / 2 - halfRad;
    const a1 = -Math.PI / 2 + halfRad;
    const edges = [a0, -Math.PI / 2 - halfRad / 3, -Math.PI / 2 + halfRad / 3, a1];

    const sectors = s?.sectors || [{ level: 0 }, { level: 0 }, { level: 0 }];
    for (let i = 0; i < 3; i++) {
      const lv = sectors[i]?.level || 0;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.arc(ox, oy, R, edges[i], edges[i + 1]);
      ctx.closePath();
      ctx.fillStyle = LEVEL_SECTOR_FILL[lv];
      ctx.fill();
    }

    if (blind > 0.05) {
      const rb = Math.min(R, blind * scale);
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.arc(ox, oy, rb, a0, a1);
      ctx.closePath();
      ctx.fillStyle = "rgba(139, 149, 168, 0.16)";
      ctx.fill();
      ctx.fillStyle = MUTED;
      ctx.font = FONT_MONO;
      ctx.textAlign = "center";
      ctx.fillText("Vùng mù", ox, oy - Math.max(16, rb * 0.48));
    }

    ctx.textAlign = "left";
    ctx.font = FONT_MONO;
    for (let r = 1; r <= Math.floor(range); r++) {
      ctx.beginPath();
      ctx.arc(ox, oy, r * scale, a0, a1);
      ctx.strokeStyle = RING;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = MUTED;
      ctx.fillText(`${r}m`, ox + r * scale * Math.cos(a1) + 5, oy + r * scale * Math.sin(a1) + 4);
    }

    for (const [d, col] of [[1.5, "rgba(239, 68, 68, 0.72)"], [2.5, "rgba(245, 158, 11, 0.55)"]]) {
      if (d <= range) {
        ctx.beginPath();
        ctx.arc(ox, oy, d * scale, a0, a1);
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    for (let i = 0; i < edges.length; i++) {
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ox + R * Math.cos(edges[i]), oy + R * Math.sin(edges[i]));
      ctx.strokeStyle = i === 0 || i === 3 ? AXIS_OUTER : AXIS_INNER;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    const scanPts = s?.scan || [];
    if (scanPts.length > 0) {
      ctx.fillStyle = "rgba(163, 169, 184, 0.65)";
      for (const [sx, sz] of scanPts) {
        if (sz > range + 0.2) continue;
        ctx.beginPath();
        ctx.arc(ox + sx * scale, oy - sz * scale, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 2D Occupancy Grid memory (dac biet hieu qua cho vung mu & ria FOV)
    const gridCells = s?.grid_cells || [];
    if (gridCells.length > 0) {
      const cellPx = Math.max(2, 0.1 * scale);
      for (const [gx, gz, occ, kind] of gridCells) {
        if (gz > range + 0.1 || Math.abs(gx) > range) continue;
        const alpha = Math.min(0.32, occ * 0.28);
        ctx.fillStyle =
          kind === 2
            ? `rgba(249, 115, 22, ${alpha})`
            : kind === 3
            ? `rgba(168, 85, 247, ${alpha})`
            : `rgba(148, 163, 184, ${alpha})`;
        ctx.fillRect(ox + gx * scale - cellPx * 0.5, oy - gz * scale - cellPx * 0.5, cellPx, cellPx);
      }
    }

    const dropPts = s?.dropoffs || [];
    if (dropPts.length > 0) {
      ctx.fillStyle = "#f97316";
      for (const [dx, dz] of dropPts) {
        if (dz > range + 0.2) continue;
        const px = ox + dx * scale;
        const py = oy - dz * scale;
        ctx.beginPath();
        ctx.moveTo(px, py - 4);
        ctx.lineTo(px - 3.5, py + 3);
        ctx.lineTo(px + 3.5, py + 3);
        ctx.closePath();
        ctx.fill();
      }
    }

    // Overhead hazards (vat can ngang dau)
    const overPts = s?.overheads || [];
    if (overPts.length > 0) {
      ctx.fillStyle = "#a855f7";
      for (const [vx, vz] of overPts) {
        if (vz > range + 0.2) continue;
        const px = ox + vx * scale;
        const py = oy - vz * scale;
        ctx.beginPath();
        ctx.moveTo(px, py - 4);
        ctx.lineTo(px + 4, py);
        ctx.lineTo(px, py + 4);
        ctx.lineTo(px - 4, py);
        ctx.closePath();
        ctx.fill();
      }
    }

    // Safe steering guidance vector
    const g = s?.guidance;
    if (g && g.action !== "stop") {
      const grad = (g.target_deg * Math.PI) / 180.0;
      const lenM = Math.min(g.clearance_m || 2.5, range * 0.85);
      const gx = ox + lenM * Math.sin(grad) * scale;
      const gy = oy - lenM * Math.cos(grad) * scale;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(gx, gy);
      ctx.strokeStyle = g.action === "caution" ? "rgba(245, 158, 11, 0.72)" : "rgba(56, 189, 248, 0.78)";
      ctx.lineWidth = 2.2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
      const ang = Math.atan2(gy - oy, gx - ox);
      ctx.beginPath();
      ctx.moveTo(gx, gy);
      ctx.lineTo(gx - 7 * Math.cos(ang - 0.45), gy - 7 * Math.sin(ang - 0.45));
      ctx.lineTo(gx - 7 * Math.cos(ang + 0.45), gy - 7 * Math.sin(ang + 0.45));
      ctx.closePath();
      ctx.fillStyle = g.action === "caution" ? "#f59e0b" : "#38bdf8";
      ctx.fill();
    }

    this.hitBoxes = [];
    for (const item of this.interp.values()) {
      item.x += 0.28 * (item.tx - item.x);
      item.z += 0.28 * (item.tz - item.z);
      const t = item.meta;
      if (item.z > range + 0.3) continue;

      const px = ox + item.x * scale;
      const py = oy - item.z * scale;
      const lv =
        t.r < 1.5 || (t.ttc != null && t.ttc < 1.5 && t.cpa < 0.6)
          ? 2
          : t.r < 2.5 || (t.ttc != null && t.ttc < 3.0 && t.cpa < 0.6)
          ? 1
          : 0;
      const col = LEVEL_HEX[lv];

      const spd = Math.hypot(t.vx || 0, t.vz || 0);
      if (spd > 0.2) {
        const ex = px + t.vx * scale;
        const ey = py - t.vz * scale;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(ex, ey);
        ctx.strokeStyle = col;
        ctx.lineWidth = 1.75;
        ctx.stroke();

        const ang = Math.atan2(ey - py, ex - px);
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 6 * Math.cos(ang - 0.45), ey - 6 * Math.sin(ang - 0.45));
        ctx.lineTo(ex - 6 * Math.cos(ang + 0.45), ey - 6 * Math.sin(ang + 0.45));
        ctx.closePath();
        ctx.fillStyle = col;
        ctx.fill();
      }

      ctx.fillStyle = col;
      ctx.strokeStyle = this.selectedId === t.id ? TEXT : BG;
      ctx.lineWidth = this.selectedId === t.id ? 2 : 1.5;
      if (t.cls === 0) {
        ctx.beginPath();
        ctx.arc(px, py, 6.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.roundRect(px - 6, py - 6, 12, 12, 3);
        ctx.fill();
        ctx.stroke();
      }

      ctx.font = FONT_UI;
      ctx.fillStyle = TEXT;
      ctx.textAlign = "left";
      const ttcStr = t.ttc != null && t.ttc < 4.0 ? ` · ${t.ttc.toFixed(1)}s` : "";
      ctx.fillText(`${t.name} ${t.r.toFixed(1)}m${ttcStr}`, px + 9, py + 4);

      this.hitBoxes.push({ px, py, track: t });
    }

    if (this.selectedId != null) {
      const sel = this.hitBoxes.find((h) => h.track.id === this.selectedId);
      if (sel) this._renderTooltip(sel);
    }

    ctx.beginPath();
    ctx.moveTo(ox, oy - 9);
    ctx.lineTo(ox - 6.5, oy + 5);
    ctx.lineTo(ox + 6.5, oy + 5);
    ctx.closePath();
    ctx.fillStyle = TEXT;
    ctx.fill();
  }
}
