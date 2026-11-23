import { RadarCanvas } from "/static/radar.js";
import { AudioEngine } from "/static/audio.js";

const $ = (id) => document.getElementById(id);
const NAMES = ["AN TOÀN", "CHÚ Ý", "NGUY HIỂM"];

const radar = new RadarCanvas($("radarCanvas"), $("radarTooltip"));
const audio = new AudioEngine();

let st = {};
let radarOnly = false;
let lastAck = 0;
let lastOkTime = Date.now();
let sse = null;

["click", "touchstart", "keydown"].forEach((ev) => {
  window.addEventListener(ev, () => audio.unlock(), { once: true, passive: true });
});

function cmd(c, q = "") {
  audio.unlock();
  return fetch(`/cmd?c=${encodeURIComponent(c)}${q}`).then(() => pollOnce());
}

function pollOnce() {
  return fetch("/state", { cache: "no-store" })
    .then((r) => r.json())
    .then((s) => handleState(s))
    .catch(() => {});
}

document.querySelectorAll(".cmd-btn").forEach((b) => {
  b.addEventListener("click", () => cmd(b.dataset.cmd));
});
$("muteBtn").addEventListener("click", () => cmd("mute"));
$("spatialBtn").addEventListener("click", () => cmd("spatial"));
$("voiceBtn").addEventListener("click", () => cmd("voice"));
$("calibBtn").addEventListener("click", () => cmd("calib"));
$("flipBtn").addEventListener("click", () => cmd("flip"));

document.querySelectorAll(".range-btn").forEach((b) => {
  b.addEventListener("click", () => cmd("radar_range", `&v=${b.dataset.range}`));
});
document.querySelectorAll(".cdist-btn").forEach((b) => {
  b.addEventListener("click", () => cmd("set_calib_dist", `&v=${b.dataset.cdist}`));
});

$("drawerToggle").addEventListener("click", () => {
  const d = $("controlsDrawer");
  const open = d.classList.toggle("hidden") === false;
  $("drawerToggle").setAttribute("aria-expanded", String(open));
  $("drawerToggle").classList.toggle("on", open);
});

$("radarOnlyBtn").addEventListener("click", () => {
  radarOnly = !radarOnly;
  const img = $("camImg");
  if (radarOnly) {
    img.src = "";
    img.classList.add("hidden");
    $("radarOnlyNotice").classList.remove("hidden");
  } else {
    $("radarOnlyNotice").classList.add("hidden");
    img.classList.remove("hidden");
    img.src = `/stream?v=cam&ts=${Date.now()}`;
  }
  $("radarOnlyBtn").classList.toggle("on", radarOnly);
});

$("camImg").addEventListener("click", (e) => {
  if (!st.calib) return;
  const img = e.currentTarget;
  const r = img.getBoundingClientRect();
  if (!img.naturalWidth || !r.width) return;
  const scale = img.naturalWidth / r.width;
  cmd("click", `&x=${Math.round((e.clientX - r.left) * scale)}&y=${Math.round((e.clientY - r.top) * scale)}`);
});

function handleState(s) {
  if (!s || s.level === undefined) return;
  st = s;
  lastOkTime = Date.now();

  const overlay = $("lostOverlay");
  const isStale = Boolean(s.stale);
  const healthAlert = Boolean(s.health?.alert);

  if (isStale) {
    $("lostTitle").textContent = "Mất tín hiệu radar — Dừng lại";
    $("lostDetail").textContent = "Luồng xử lý bị gián đoạn quá 1.5 giây hoặc mất camera";
    overlay.classList.remove("hidden");
    $("liveDot").className = "dot danger";
  } else if (healthAlert) {
    $("lostTitle").textContent = "Camera bị che hoặc thiếu sáng";
    $("lostDetail").textContent = `Độ sáng ${s.health.brightness} · Độ nét ${s.health.sharpness}`;
    overlay.classList.remove("hidden");
    $("liveDot").className = "dot warn";
  } else {
    overlay.classList.add("hidden");
    $("liveDot").className = "dot";
  }

  const lv = s.level ?? 0;
  $("levelCard").className = `card level l${lv}`;
  $("levelText").textContent = NAMES[lv];
  document.body.classList.toggle("alarm-ring", lv === 2);

  const hasDrop = (s.dropoffs || []).length > 0;
  const hasOver = (s.overheads || []).length > 0;
  const minTtc = Math.min(...(s.sectors || []).map((x) => x.ttc ?? 999));
  const g = s.guidance;
  let guideTxt = "";
  if (g) {
    if (g.action === "left") guideTxt = `Lách trái ${Math.abs(g.target_deg).toFixed(0)}° (${g.clearance_m.toFixed(1)}m)`;
    else if (g.action === "right") guideTxt = `Lách phải ${Math.abs(g.target_deg).toFixed(0)}° (${g.clearance_m.toFixed(1)}m)`;
    else if (g.action === "stop") guideTxt = "Đường hẹp · Dừng";
    else if (g.action === "caution") guideTxt = `Chậm (${g.clearance_m.toFixed(1)}m)`;
    else guideTxt = `Đi thẳng (${g.clearance_m.toFixed(1)}m)`;
  }
  $("levelSub").innerHTML =
    (guideTxt ? `<div>${guideTxt}</div>` : "") +
    (minTtc < 99 ? `<div>TTC ${minTtc.toFixed(1)}s</div>` : "") +
    (hasDrop ? `<div style="color:var(--dropoff)">Bậc thang</div>` : "") +
    (hasOver ? `<div style="color:#c084fc">Ngang đầu</div>` : "");

  (s.sectors || []).forEach((x, i) => {
    const el = $("s" + i);
    el.className = `card sector l${x.level}`;
    el.querySelector(".dist").innerHTML = x.d == null ? "–" : `${x.d.toFixed(1)}<small>m</small>`;
    const badge = el.querySelector(".s-badge");
    if (x.dropoff) {
      badge.textContent = "BẬC";
      badge.className = "s-badge badge drop";
    } else if (x.overhead) {
      badge.textContent = "ĐẦU";
      badge.className = "s-badge badge over";
    } else if (x.ttc != null && x.ttc < 4.0) {
      badge.textContent = `${x.ttc.toFixed(1)}s`;
      badge.className = "s-badge badge ttc";
    } else {
      badge.className = "s-badge hidden";
    }
  });

  $("computeBadge").textContent = `· ${(s.compute_backend || "py").toUpperCase()}`;
  $("aiBadge").textContent = s.ai_backend ? `· ${s.ai_backend}` : "";
  $("objsBadge").textContent = `${s.n ?? 0} vật`;
  $("fpsBadge").textContent = `${(s.fps ?? 0).toFixed(1)} FPS`;

  $("pitchVal").textContent = `${(s.pitch ?? 0).toFixed(1)}°`;
  $("heightVal").textContent = `${(s.height ?? 1.5).toFixed(2)} m`;
  $("hfovVal").textContent = `${(s.hfov ?? 70).toFixed(0)}°`;
  $("blindVal").textContent = `${(s.blind ?? 0).toFixed(1)} m`;

  $("tiltVal").textContent =
    s.tilt == null
      ? "Cố định"
      : `${s.tilt >= 0 ? "+" : ""}${s.tilt.toFixed(1)}° / ${s.roll.toFixed(1)}°`;
  $("sensorDot").className = `dot ${s.tilt == null ? "muted" : ""}`;

  const d = s.depth;
  $("depthVal").textContent = !d ? "Tắt" : d.ready ? `${d.ms}ms (${d.anchors})` : "Chờ neo";
  $("depthDot").className = `dot ${!d ? "muted" : d.ready ? "" : "warn"}`;

  $("healthVal").textContent = healthAlert ? "Bị che" : "OK";
  $("healthDot").className = `dot ${healthAlert ? "danger" : ""}`;

  const tm = s.timings;
  if (tm) {
    $("perfVal").textContent = `Y ${tm.yolo_ms.toFixed(0)} · P ${tm.pose_ms.toFixed(0)} · D ${tm.depth_ms.toFixed(0)} · C ${tm.core_ms.toFixed(1)} ms`;
  }
  const ph = s.pitch_hint;
  $("pitchHintVal").textContent = ph ? `Gợi ý pitch ~${ph.suggested_deg.toFixed(1)}°` : "";

  $("muteBtn").classList.toggle("on", !s.muted);
  $("muteLabel").textContent = s.muted ? "Tắt tiếng" : "Âm thanh";
  $("spatialBtn").classList.toggle("on", Boolean(s.spatial_audio));
  $("voiceBtn").classList.toggle("on", Boolean(s.voice_enabled));
  $("flipBtn").classList.toggle("on", Boolean(s.flip));
  $("calibBtn").classList.toggle("warn-on", Boolean(s.calib));
  $("calibLabel").textContent = s.calib ? "Hủy" : "Hiệu chỉnh";

  $("camImg").classList.toggle("calib", Boolean(s.calib));
  $("calibHint").classList.toggle("hidden", !s.calib);
  $("calibHintDist").textContent = (s.calib_dist ?? 2.0).toFixed(1);

  if (s.calib_ack && s.calib_ack > lastAck) {
    lastAck = s.calib_ack;
    $("calibToast").classList.remove("hidden");
    setTimeout(() => $("calibToast").classList.add("hidden"), 2500);
  }

  document.querySelectorAll(".range-btn").forEach((b) => {
    b.classList.toggle("on", Number(b.dataset.range) === Number(s.radar_range));
  });
  document.querySelectorAll(".cdist-btn").forEach((b) => {
    b.classList.toggle("on", Math.abs(Number(b.dataset.cdist) - Number(s.calib_dist)) < 0.05);
  });

  radar.setState(s);
  audio.syncFlags(s);
  audio.processState(s);
}

function connectSSE() {
  if (sse) sse.close();
  sse = new EventSource("/events");
  sse.onmessage = (ev) => {
    try {
      handleState(JSON.parse(ev.data));
    } catch {
      // Ignore malformed SSE payload
    }
  };
  sse.onerror = () => {
    sse.close();
    setTimeout(connectSSE, 1000);
  };
}

setInterval(() => {
  if (Date.now() - lastOkTime > 2000) {
    $("lostTitle").textContent = "Mất kết nối máy chủ — Dừng lại";
    $("lostDetail").textContent = "Đang thử kết nối lại qua SSE";
    $("lostOverlay").classList.remove("hidden");
    $("liveDot").className = "dot danger";
    pollOnce();
  }
}, 500);

connectSSE();
pollOnce();
