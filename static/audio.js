// Web Spatial Audio (StereoPannerNode) + Vietnamese Web Speech API + Mobile Haptics

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.muted = false;
    this.spatial = true;
    this.voice = false;
    this.lastBeep = 0;
    this.lastSpeech = 0;
    this.lastSpeechKey = "";
    this.lastBuzz = 0;
  }

  unlock() {
    if (!this.ctx && (window.AudioContext || window.webkitAudioContext)) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx();
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume().catch(() => {});
    }
  }

  syncFlags({ muted, spatial_audio, voice_enabled }) {
    this.muted = Boolean(muted);
    this.spatial = Boolean(spatial_audio);
    this.voice = Boolean(voice_enabled);
  }

  processState(s) {
    const lv = s.level ?? 0;
    const now = performance.now();

    // Mobile Haptics
    if (!this.muted && navigator.vibrate) {
      if (lv === 2 && now - this.lastBuzz > 450) {
        navigator.vibrate([120, 60, 120]);
        this.lastBuzz = now;
      } else if (lv === 1 && now - this.lastBuzz > 1400) {
        navigator.vibrate(70);
        this.lastBuzz = now;
      }
    }

    if (this.muted || lv === 0) return;

    // Determine worst sector & panning (-0.85 left, 0 center, +0.85 right)
    const ws = s.worst_sector ?? 1;
    const panVal = this.spatial ? (ws === 0 ? -0.85 : ws === 2 ? 0.85 : 0.0) : 0.0;
    const sec = (s.sectors || [])[ws] || {};
    const dist = sec.d ?? 2.0;

    // Cadence & frequency scale as distance/TTC drops
    const intervalMs = lv === 2 ? Math.max(180, Math.min(450, dist * 240)) : 750;
    const freq = lv === 2 ? Math.max(900, 1650 - dist * 220) : 780;

    if (this.ctx && this.ctx.state === "running" && now - this.lastBeep >= intervalMs) {
      this._beep(freq, lv === 2 ? 0.09 : 0.07, panVal);
      this.lastBeep = now;
    }

    // Vietnamese voice alerts (cooldown 3.5s)
    if (this.voice && "speechSynthesis" in window && now - this.lastSpeech > 3500) {
      const msg = this._buildSpeech(s, ws, lv);
      if (msg && (msg !== this.lastSpeechKey || now - this.lastSpeech > 6000)) {
        this._speak(msg);
        this.lastSpeech = now;
        this.lastSpeechKey = msg;
      }
    }
  }

  _beep(freq, durSec, pan) {
    try {
      const t0 = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(freq, t0);
      gain.gain.setValueAtTime(0.001, t0);
      gain.gain.exponentialRampToValueAtTime(0.18, t0 + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, t0 + durSec);

      if (this.ctx.createStereoPanner) {
        const panner = this.ctx.createStereoPanner();
        panner.pan.setValueAtTime(pan, t0);
        osc.connect(gain).connect(panner).connect(this.ctx.destination);
      } else {
        osc.connect(gain).connect(this.ctx.destination);
      }
      osc.start(t0);
      osc.stop(t0 + durSec + 0.02);
    } catch {
      // Ignore audio errors when context blocked
    }
  }

  _buildSpeech(s, ws, lv) {
    if (s.health?.alert) return "Camera bị che hoặc thiếu sáng";
    const sideNames = ["bên trái", "phía trước", "bên phải"];
    const side = sideNames[ws] || "phía trước";
    if ((s.overheads || []).length > 0 && s.sectors?.[ws]?.overhead) {
      return `Cẩn thận vật cản ngang đầu ${side}`;
    }
    if ((s.dropoffs || []).length > 0 && s.sectors?.[ws]?.dropoff) {
      return `Cẩn thận bậc thang ${side}`;
    }
    if (lv >= 1 && ws === 1) {
      const act = s.guidance?.action;
      if (act === "left") return "Vật cản phía trước, lách sang trái";
      if (act === "right") return "Vật cản phía trước, lách sang phải";
      if (act === "stop") return "Vật cản chặn phía trước, dừng lại";
    }
    const sec = s.sectors?.[ws];
    if (lv >= 1 && sec?.d != null) {
      const closest = (s.tracks || []).slice().sort((a, b) => a.r - b.r)[0];
      const label = closest?.cls === 0 ? "Người" : "Vật cản";
      return `${label} ${side}, ${sec.d.toFixed(1)} mét`;
    }
    return null;
  }

  _speak(text) {
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "vi-VN";
      u.rate = 1.15;
      window.speechSynthesis.speak(u);
    } catch {
      // Speech API unsupported
    }
  }
}
