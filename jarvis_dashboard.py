"""Local read-only web dashboard for JARVIS runtime state."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
import shutil
import subprocess
import threading
import webbrowser


HOST = "localhost"
PORT = 8766
URL = f"http://{HOST}:{PORT}/"
_lock = threading.Lock()
_server = None
_settings_handler = None
_face_service = None
_twilio_service = None
_performance_lock = threading.Lock()
_previous_cpu_times = None
_state = {
    "status": "starting",
    "activity": "Starting JARVIS",
    "model": "qwen3.5:4b",
    "mode": "FAST",
    "tools": [],
    "telegram": "not-configured",
    "twilio": {},
    "settings": {},
    "facial": {"status": "unavailable", "identity": None, "profile_info": "",
                "profiles": [], "message": "Face recognition is not configured."},
    "messages": [],
    "events": [],
}


def set_state(**values):
    with _lock:
        _state.update(values)


def add_message(role, content):
    item = {
        "time": datetime.now().astimezone().strftime("%H:%M:%S"),
        "role": str(role)[:24],
        "content": str(content)[:1200],
    }
    with _lock:
        _state["messages"].append(item)
        del _state["messages"][:-40]


def add_event(kind, detail):
    item = {
        "time": datetime.now().astimezone().strftime("%H:%M:%S"),
        "kind": str(kind)[:40],
        "detail": str(detail)[:400],
    }
    with _lock:
        _state["events"].append(item)
        del _state["events"][:-30]


def snapshot():
    with _lock:
        return json.loads(json.dumps(_state))


def performance_snapshot():
    global _previous_cpu_times
    result = {"cpu_percent": None, "memory_percent": None, "memory_used": None,
              "memory_total": None, "disk_percent": None, "disk_used": None,
              "disk_total": None, "load_average": None, "uptime_seconds": None}
    try:
        with open("/proc/stat", encoding="ascii") as stat_file:
            cpu_line = stat_file.readline().split()
        if cpu_line[0] == "cpu" and len(cpu_line) >= 8:
            cpu_times = [int(value) for value in cpu_line[1:9]]
            with _performance_lock:
                previous = _previous_cpu_times
                _previous_cpu_times = cpu_times
            if previous is not None:
                total_delta = sum(cpu_times) - sum(previous)
                idle_delta = sum(cpu_times[3:5]) - sum(previous[3:5])
                if total_delta > 0:
                    result["cpu_percent"] = round(
                        100 * (total_delta - idle_delta) / total_delta, 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        memory = {}
        with open("/proc/meminfo", encoding="ascii") as mem_file:
            for line in mem_file:
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    memory[key] = int(value.split()[0]) * 1024
        total = memory["MemTotal"]
        available = memory["MemAvailable"]
        result.update(memory_total=total, memory_used=total - available,
                      memory_percent=round(100 * (total - available) / total, 1))
    except (OSError, ValueError, KeyError, ZeroDivisionError):
        pass
    try:
        disk = shutil.disk_usage("/")
        result.update(disk_total=disk.total, disk_used=disk.used,
                      disk_percent=round(100 * disk.used / disk.total, 1))
    except (OSError, ZeroDivisionError):
        pass
    try:
        result["load_average"] = [round(value, 2) for value in os.getloadavg()]
    except (OSError, AttributeError):
        pass
    try:
        with open("/proc/uptime", encoding="ascii") as uptime_file:
            result["uptime_seconds"] = int(float(uptime_file.read().split()[0]))
    except (OSError, ValueError, IndexError):
        pass
    return result


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' http://localhost:8081">
  <title>JARVIS Dashboard</title>
  <style>
    :root { color-scheme:dark; --bg:#0b1018; --sidebar:#0c141f; --panel:#121b28; --line:#243247; --ink:#e8f0fa; --muted:#8da2bb; --cyan:#64d5e7; --gold:#f2c879; }
    * { box-sizing:border-box }
    body { margin:0; min-height:100vh; background:radial-gradient(ellipse at top,#17273b,var(--bg) 58%); color:var(--ink); font:15px/1.5 system-ui,sans-serif }
    .app-shell { min-height:100vh; display:grid; grid-template-columns:244px minmax(0,1fr) }
    .sidebar { position:sticky; top:0; height:100vh; display:flex; flex-direction:column; gap:28px; padding:24px 15px 18px; background:var(--sidebar); border-right:1px solid var(--line) }
    .brand { display:flex; align-items:center; gap:11px; padding:0 8px }
    .brand-mark { display:grid; place-items:center; width:39px; height:39px; border:1px solid #3b6b7b; border-radius:12px; background:#102733; color:var(--cyan); font-weight:800; font-size:21px }
    .brand-name { font-size:17px; font-weight:750; letter-spacing:.14em }
    .brand-caption,.eyebrow { color:var(--muted); font-size:10px; letter-spacing:.14em; text-transform:uppercase }
    .nav-menu { display:grid; gap:6px }
    .nav-link { display:flex; align-items:center; gap:12px; padding:10px 11px; border:1px solid transparent; border-radius:10px; color:#b6c4d5; text-decoration:none; transition:background .15s,border-color .15s,color .15s }
    .nav-link:hover { color:var(--ink); background:#131f2c }
    .nav-link[aria-current="page"] { color:var(--cyan); background:#10242e; border-color:#24414d }
    .nav-icon { width:24px; color:#8094aa; font-size:10px; font-weight:700; letter-spacing:.06em }
    .nav-link[aria-current="page"] .nav-icon { color:var(--cyan) }
    .sidebar-foot { display:flex; align-items:center; gap:9px; margin-top:auto; padding:12px 9px 0; border-top:1px solid var(--line); color:var(--muted); font-size:12px }
    .local-dot { width:8px; height:8px; border-radius:50%; background:#64d5a4; box-shadow:0 0 10px #64d5a466 }
    .content { width:min(1380px,100%); margin:0 auto; padding:30px clamp(18px,3.3vw,46px) 20px }
    .topbar { display:flex; align-items:center; justify-content:space-between; gap:20px; margin-bottom:25px; padding-bottom:20px; border-bottom:1px solid #243247aa }
    .topbar h1 { margin:3px 0 2px; font-size:clamp(25px,3vw,34px); letter-spacing:.01em }
    .subtitle,.muted { color:var(--muted) }
    .subtitle { margin:0; font-size:13px }
    .runtime { display:grid; justify-items:end; gap:5px }
    .runtime-label { color:var(--muted); font-size:10px; letter-spacing:.12em; text-transform:uppercase }
    .badge { border:1px solid #2c5260; border-radius:999px; padding:6px 12px; color:var(--cyan); background:#0e1824; text-transform:uppercase; font-size:11px; letter-spacing:.1em }
    .page[hidden] { display:none }
    .page { animation:page-in .18s ease-out }
    @keyframes page-in { from { opacity:.55; transform:translateY(3px) } to { opacity:1; transform:translateY(0) } }
    .page-intro { margin:-7px 0 16px; color:var(--muted); font-size:13px }
    .grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin-top:14px }
    .performance-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px }
    .performance-card { padding:14px; border:1px solid var(--line); border-radius:10px; background:#0e1722 }
    .performance-detail { color:var(--muted); font-size:12px; margin-top:4px; overflow-wrap:anywhere }
    .performance-panel { margin-top:14px }
    .card,.panel { background:linear-gradient(145deg,#162333,var(--panel)); border:1px solid var(--line); border-radius:15px; box-shadow:0 12px 30px #0003 }
    .panel { padding:20px; min-height:0 }
    .overview-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px }
    .overview-head h2 { margin:0 }
    .tool-count { color:var(--muted); font-size:12px }
    .tools { display:flex; flex-wrap:wrap; gap:8px }
    .tool { border:1px solid #2c4359; border-radius:999px; background:#0e1824; color:var(--cyan); padding:5px 10px; font-size:12px }
    .card { padding:18px 20px; min-height:106px }
    .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.12em }
    .value { margin-top:8px; font-size:20px; font-weight:650; overflow-wrap:anywhere }
    .settings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px }
    .setting { display:grid; gap:5px; color:var(--muted); font-size:12px }
    .setting input,.setting select,.setting textarea { width:100%; border:1px solid var(--line); border-radius:8px; padding:9px; background:#0b141f; color:var(--ink); font:inherit }
    .setting input:focus,.setting select:focus,.setting textarea:focus,.control-button:focus-visible,.nav-link:focus-visible { outline:2px solid var(--cyan); outline-offset:2px }
    .setting.check { display:flex; align-items:center; gap:8px; padding-top:22px }
    .setting.check input { width:auto }
    .control-button { border:1px solid #3c627d; border-radius:8px; padding:9px 13px; color:var(--ink); background:#173048; cursor:pointer; font:inherit }
    .control-button:hover { background:#1c3a54 }
    .control-button.danger { border-color:#74434b; background:#351c24 }
    .control-row { display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-top:14px }
    .control-message { color:var(--muted); font-size:12px }
    .face-summary { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; margin-bottom:12px }
    .face-value { border:1px solid var(--line); background:#0e1722; padding:10px; border-radius:9px; overflow-wrap:anywhere }
    .profiles { display:grid; gap:8px; margin-top:12px }
    .profile { display:flex; justify-content:space-between; gap:12px; align-items:center; padding:9px 11px; background:#0e1722; border-radius:9px }
    .profile-info { color:var(--muted); font-size:12px; white-space:pre-wrap }
    .stream-frame { width:100%; aspect-ratio:16/9; display:grid; place-items:center; overflow:hidden; border-radius:10px; background:#080d14; border:1px solid var(--line) }
    .stream-frame img { display:block; width:100%; height:100%; object-fit:contain }
    .stream-note { color:var(--muted); font-size:12px; margin-top:8px }
    h2 { font-size:13px; text-transform:uppercase; letter-spacing:.13em; margin:0 0 14px; color:var(--gold) }
    .list { display:grid; gap:10px; max-height:calc(100vh - 260px); min-height:180px; overflow:auto }
    .entry { border-left:2px solid var(--cyan); padding:8px 11px; background:#0e1722; border-radius:0 9px 9px 0; overflow-wrap:anywhere }
    .entry.assistant { border-color:var(--gold) }
    .meta { color:var(--muted); font-size:11px; margin-bottom:3px; text-transform:uppercase; letter-spacing:.08em }
    .empty { color:var(--muted); padding:10px 0 }
    footer { color:var(--muted); font-size:11px; text-align:right; margin-top:24px }
    @media(max-width:850px) {
      .app-shell { display:block }
      .sidebar { position:static; height:auto; gap:14px; padding:14px 16px 10px; border-right:0; border-bottom:1px solid var(--line) }
      .brand { padding:0 2px }
      .nav-menu { display:flex; gap:6px; overflow-x:auto; padding:2px 0 5px }
      .nav-link { flex:0 0 auto; padding:8px 10px }
      .sidebar-foot { display:none }
      .content { padding-top:22px }
    }
    @media(max-width:650px) {
      .content { padding:18px 14px }
      .topbar { align-items:flex-start; margin-bottom:18px }
      .runtime { padding-top:4px }
      .grid,.face-summary,.settings-grid { grid-template-columns:1fr }
      .performance-grid { grid-template-columns:repeat(2,minmax(0,1fr)) }
      .profile { align-items:flex-start; flex-direction:column }
      .profile .control-row { margin-top:4px }
      .list { max-height:none }
      .overview-head { align-items:flex-start; flex-direction:column }
    }
    @media(max-width:420px) {
      .performance-grid { grid-template-columns:1fr }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-mark" aria-hidden="true">J</div><div><div class="brand-name">JARVIS</div><div class="brand-caption">Local assistant</div></div></div>
      <nav class="nav-menu" aria-label="Dashboard pages">
        <a class="nav-link" href="#overview" data-page="overview" aria-current="page"><span class="nav-icon" aria-hidden="true">01</span><span>Overview</span></a>
        <a class="nav-link" href="#conversation" data-page="conversation"><span class="nav-icon" aria-hidden="true">02</span><span>Conversation</span></a>
        <a class="nav-link" href="#activity" data-page="activity"><span class="nav-icon" aria-hidden="true">03</span><span>Activity</span></a>
        <a class="nav-link" href="#camera" data-page="camera"><span class="nav-icon" aria-hidden="true">04</span><span>Camera</span></a>
        <a class="nav-link" href="#recognition" data-page="recognition"><span class="nav-icon" aria-hidden="true">05</span><span>Face recognition</span></a>
        <a class="nav-link" href="#settings" data-page="settings"><span class="nav-icon" aria-hidden="true">06</span><span>Settings</span></a>
      </nav>
      <div class="sidebar-foot"><span class="local-dot" aria-hidden="true"></span><span>Running on this computer</span></div>
    </aside>
    <main class="content" id="main-content">
      <header class="topbar">
        <div><div class="eyebrow">JARVIS control center</div><h1 id="page-title">Overview</h1><p id="page-subtitle" class="subtitle">Live runtime status and available capabilities.</p></div>
        <div class="runtime"><span class="runtime-label">Runtime status</span><div id="status" class="badge">Starting</div></div>
      </header>

      <section class="page" id="page-overview" data-title="Overview" data-subtitle="Live runtime status and available capabilities.">
        <section class="panel">
          <div class="overview-head"><h2>System overview</h2><div><span id="tool-count" class="tool-count">Loading tools…</span><span id="telegram-status" class="tool-count"> · Telegram: not configured</span></div></div>
          <div id="tools" class="tools"><span class="muted">Waiting for JARVIS startup…</span></div>
        </section>
        <section class="grid">
          <article class="card"><div class="label">Active model</div><div id="model" class="value">qwen3.5:4b</div></article>
          <article class="card"><div class="label">Effort mode</div><div id="mode" class="value">FAST</div></article>
          <article class="card"><div class="label">Current activity</div><div id="activity" class="value">Starting JARVIS</div></article>
        </section>
        <article class="panel performance-panel">
          <div class="overview-head"><h2>Computer performance</h2><span class="tool-count">Live · refreshes every 1.2 seconds</span></div>
          <div class="performance-grid">
            <div class="performance-card"><div class="label">CPU usage</div><div id="perf-cpu" class="value">Sampling…</div><div class="performance-detail">Overall processor utilization</div></div>
            <div class="performance-card"><div class="label">Memory usage</div><div id="perf-memory" class="value">Loading…</div><div id="perf-memory-detail" class="performance-detail"></div></div>
            <div class="performance-card"><div class="label">Disk usage · /</div><div id="perf-disk" class="value">Loading…</div><div id="perf-disk-detail" class="performance-detail"></div></div>
            <div class="performance-card"><div class="label">System load · 1/5/15 min</div><div id="perf-load" class="value">Loading…</div><div id="perf-uptime" class="performance-detail"></div></div>
          </div>
        </article>
      </section>

      <section class="page" id="page-conversation" data-title="Conversation" data-subtitle="Review recent messages exchanged with JARVIS." hidden>
        <p class="page-intro">Recent conversation messages are kept in memory while JARVIS is running.</p>
        <article class="panel"><h2>Recent conversation</h2><div id="messages" class="list"><div class="empty">Waiting for conversation…</div></div></article>
      </section>

      <section class="page" id="page-activity" data-title="Activity" data-subtitle="See what JARVIS and its tools have been doing." hidden>
        <p class="page-intro">Runtime events, service updates, and tool activity appear here.</p>
        <article class="panel"><h2>Recent activity</h2><div id="events" class="list"><div class="empty">No activity yet.</div></div></article>
      </section>

      <section class="page" id="page-camera" data-title="Camera" data-subtitle="View the local Motion camera stream." hidden>
        <article class="panel">
          <h2>Motion camera</h2>
          <div class="stream-frame"><img src="http://localhost:8081/0/stream" alt="Live motion camera stream" onerror="document.getElementById('stream-status').textContent='Stream unavailable at localhost:8081/0/stream'"></div>
          <div id="stream-status" class="stream-note">Live stream: localhost:8081/0/stream</div>
        </article>
      </section>

      <section class="page" id="page-recognition" data-title="Face recognition" data-subtitle="Manage local recognition, profiles, and permissions." hidden>
        <article class="panel">
          <h2>Facial recognition · Local</h2>
          <div class="face-summary">
            <div class="face-value"><div class="label">Status</div><div id="face-status" class="value">Unavailable</div></div>
            <div class="face-value"><div class="label">Recognized user</div><div id="face-identity" class="value">None</div></div>
            <div class="face-value"><div class="label">Enrolled profiles</div><div id="face-count" class="value">0</div></div>
          </div>
          <div id="face-message" class="muted">Face recognition is not configured.</div>
          <div id="face-profile-info" class="profile-info"></div>
          <form id="enroll-form">
            <div class="settings-grid">
              <label class="setting">Name<input name="name" maxlength="80" required placeholder="Profile name"></label>
              <label class="setting">Profile details<input name="info" maxlength="500" placeholder="Optional user information"></label>
              <label class="setting">Permissions<select name="role"><option value="guest">Guest · limited tools</option><option value="user" selected>User · standard tools</option><option value="admin">Admin · elevated tools require terminal confirmation and sudo</option></select></label>
            </div>
            <div class="control-row"><button class="control-button" type="submit">Enroll from camera</button><button class="control-button danger" id="cancel-enrollment" type="button">Cancel enrollment</button><span id="enroll-message" class="control-message"></span></div>
          </form>
          <div class="control-row"><button class="control-button" id="face-start" type="button">Start recognition</button><button class="control-button danger" id="face-stop" type="button">Stop recognition</button></div>
          <div id="profiles" class="profiles"></div>
          <div class="stream-note">Profiles and samples stay under ~/.config/jarvis/faces. Face matches only select a permissions profile; they do not prove who spoke. Admin commands still require typed approval and OS sudo authentication.</div>
        </article>
      </section>

      <section class="page" id="page-settings" data-title="Settings" data-subtitle="Adjust JARVIS behavior and runtime preferences." hidden>
        <article class="panel">
          <h2>Assistant settings</h2>
          <form id="settings-form">
            <div class="settings-grid">
              <label class="setting">Active model mode<select name="mode"><option value="fast">Fast · 4B</option><option value="normal">Normal · 9B</option><option value="deep">Deep · 14B</option></select></label>
              <label class="setting">Wake phrase<input name="wake_word" maxlength="50" required></label>
              <label class="setting">TTS voice<select name="tts_voice"><option>bm_george</option><option>bm_fable</option><option>bm_lewis</option><option>bm_daniel</option></select></label>
              <label class="setting">Speech speed<input name="tts_speed" type="number" min="0.5" max="1.5" step="0.05"></label>
              <label class="setting">Motion announcement cooldown (seconds)<input name="motion_cooldown" type="number" min="10" max="3600" step="10"></label>
              <label class="setting">Default weather location<input name="default_location" maxlength="100"></label>
              <label class="setting">Speech recognition model (restart to apply)<select name="whisper_model"><option>tiny.en</option><option>base.en</option><option>small.en</option></select></label>
              <label class="setting">Model keep-alive<select name="keep_alive"><option>30s</option><option>1m</option><option>5m</option><option>10m</option><option>30m</option><option>1h</option></select></label>
              <label class="setting">Maximum agent tool rounds<input name="max_agent_steps" type="number" min="1" max="10" step="1"></label>
              <label class="setting">Conversation history turns<input name="max_history" type="number" min="2" max="20" step="1"></label>
              <label class="setting">Remembered fact limit<input name="max_facts" type="number" min="10" max="200" step="10"></label>
              <label class="setting">Motion event polling (seconds)<input name="motion_poll" type="number" min="1" max="30" step="1"></label>
              <label class="setting">Silence before command ends (seconds)<input name="command_silence_seconds" type="number" min="0.5" max="5" step="0.25"></label>
              <label class="setting">Maximum voice command duration (seconds)<input name="command_max_seconds" type="number" min="5" max="60" step="1"></label>
              <label class="setting">Away time before goodbye (seconds)<input name="presence_leave_seconds" type="number" min="30" max="600" step="10"></label>
              <label class="setting check"><input name="wake_word_enabled" type="checkbox"> Voice activation enabled</label>
              <label class="setting check"><input name="motion_announce" type="checkbox"> Motion announcements enabled</label>
              <label class="setting check"><input name="presence_announce" type="checkbox"> Arrival and departure announcements enabled</label>
            </div>
            <div class="control-row"><button class="control-button" type="submit">Save settings</button><span id="settings-message" class="control-message"></span></div>
          </form>
        </article>
        <article class="panel" style="margin-top:14px">
          <h2>Twilio SMS and VoIP voice</h2>
          <p class="stream-note">Credentials stay in ~/.config/jarvis/twilio.json. Inbound webhooks require a public HTTPS tunnel to localhost:8767.</p>
          <form id="twilio-form">
            <div class="settings-grid">
              <label class="setting">Account SID<input name="account_sid" autocomplete="off" placeholder="AC…"></label>
              <label class="setting">Auth Token<input name="auth_token" type="password" autocomplete="new-password" placeholder="Leave blank to keep saved token"></label>
              <label class="setting">Twilio phone number<input name="phone_number" placeholder="+15551234567"></label>
              <label class="setting">Public HTTPS webhook origin<input name="webhook_url" placeholder="https://your-public-host.example"></label>
              <label class="setting">Allowed phone numbers · E.164<textarea name="allowed_numbers" rows="3" placeholder="+15551234567, one number per line"></textarea></label>
              <label class="setting check"><input name="enabled" type="checkbox"> Enable Twilio</label>
              <label class="setting check"><input name="inbound_sms" type="checkbox"> Allow inbound SMS</label>
              <label class="setting check"><input name="inbound_voice" type="checkbox"> Allow inbound voice</label>
              <label class="setting check"><input name="outbound_sms" type="checkbox"> Allow outbound SMS</label>
              <label class="setting check"><input name="outbound_voice" type="checkbox"> Allow outbound voice calls</label>
            </div>
            <div id="twilio-webhook-paths" class="stream-note"></div>
            <div class="control-row"><button class="control-button" type="submit">Save Twilio settings</button><span id="twilio-message" class="control-message"></span></div>
          </form>
        </article>
      </section>
      <footer>Served locally at localhost:8766 · refreshes automatically</footer>
    </main>
  </div>
  <script>
    function selectPage(id){
      const page=document.getElementById(`page-${id}`)||document.getElementById("page-overview");
      for(const item of document.querySelectorAll(".page"))item.hidden=item!==page;
      for(const link of document.querySelectorAll(".nav-link")){
        if(link.dataset.page===page.id.slice(5))link.setAttribute("aria-current","page");
        else link.removeAttribute("aria-current");
      }
      document.getElementById("page-title").textContent=page.dataset.title;
      document.getElementById("page-subtitle").textContent=page.dataset.subtitle;
      document.title=`${page.dataset.title} · JARVIS`;
    }
    function syncPageFromHash(){selectPage(window.location.hash.slice(1)||"overview");}
    window.addEventListener("hashchange",syncPageFromHash);
    syncPageFromHash();
    const put=(id,value)=>document.getElementById(id).textContent=value||"—";
    function renderList(id,items,kind){
      const root=document.getElementById(id); root.replaceChildren();
      if(!items||!items.length){const e=document.createElement("div");e.className="empty";e.textContent=id==="messages"?"Waiting for conversation…":"No activity yet.";root.append(e);return;}
      for(const item of [...items].reverse()){
        const row=document.createElement("div");row.className="entry"+(item.role==="assistant"?" assistant":"");
        const meta=document.createElement("div");meta.className="meta";meta.textContent=kind==="message"?`${item.time} · ${item.role}`:`${item.time} · ${item.kind}`;
        const body=document.createElement("div");body.textContent=kind==="message"?item.content:item.detail;
        row.append(meta,body);root.append(row);
      }
    }
    function renderTools(items){
      const root=document.getElementById("tools");root.replaceChildren();
      const tools=Array.isArray(items)?items:[];
      document.getElementById("tool-count").textContent=`${tools.length} agent tool${tools.length===1?"":"s"} loaded`;
      if(!tools.length){const e=document.createElement("span");e.className="muted";e.textContent="No agent tools reported.";root.append(e);return;}
      for(const name of tools){const badge=document.createElement("span");badge.className="tool";badge.textContent=name;root.append(badge);}
    }
    function formatBytes(value){
      if(value===null||value===undefined)return "Unavailable";
      const units=["B","KB","MB","GB","TB"];let size=value,index=0;
      while(size>=1024&&index<units.length-1){size/=1024;index++;}
      return `${size.toFixed(index===0?0:1)} ${units[index]}`;
    }
    function formatUptime(seconds){
      if(seconds===null||seconds===undefined)return "Uptime unavailable";
      const days=Math.floor(seconds/86400),hours=Math.floor(seconds%86400/3600),minutes=Math.floor(seconds%3600/60);
      return `Uptime ${days}d ${hours}h ${minutes}m`;
    }
    function renderPerformance(p){
      if(!p)return;
      put("perf-cpu",p.cpu_percent===null?"Sampling…":`${p.cpu_percent}%`);
      put("perf-memory",p.memory_percent===null?"Unavailable":`${p.memory_percent}%`);
      put("perf-memory-detail",`${formatBytes(p.memory_used)} of ${formatBytes(p.memory_total)}`);
      put("perf-disk",p.disk_percent===null?"Unavailable":`${p.disk_percent}%`);
      put("perf-disk-detail",`${formatBytes(p.disk_used)} of ${formatBytes(p.disk_total)}`);
      put("perf-load",p.load_average?p.load_average.join(" · "):"Unavailable");
      put("perf-uptime",formatUptime(p.uptime_seconds));
    }
    function renderSettings(s){
      if(!s||settingsDirty)return;const form=document.getElementById("settings-form");
      for(const key of ["mode","wake_word","tts_voice","tts_speed","motion_cooldown","default_location","whisper_model","keep_alive","max_agent_steps","max_history","max_facts","motion_poll","command_silence_seconds","command_max_seconds","presence_leave_seconds"]){if(s[key]!==undefined)form.elements[key].value=s[key];}
      for(const key of ["wake_word_enabled","motion_announce","presence_announce"]){if(s[key]!==undefined)form.elements[key].checked=!!s[key];}
    }
    let settingsDirty=false;
    document.getElementById("settings-form").addEventListener("input",()=>settingsDirty=true);
    document.getElementById("settings-form").addEventListener("change",()=>settingsDirty=true);
    let twilioDirty=false;
    function renderTwilio(t){
      if(!t||twilioDirty)return;
      const form=document.getElementById("twilio-form");
      for(const key of ["account_sid","phone_number","webhook_url"]){form.elements[key].value=t[key]||"";}
      form.elements.auth_token.value="";
      form.elements.auth_token.placeholder=t.auth_token_configured?"Saved · leave blank to keep":"Enter Twilio Auth Token";
      form.elements.allowed_numbers.value=(t.allowed_numbers||[]).join("\\n");
      for(const key of ["enabled","inbound_sms","inbound_voice","outbound_sms","outbound_voice"]){form.elements[key].checked=!!t[key];}
      put("twilio-webhook-paths",`Webhook endpoints: ${t.webhook_url||"set a public HTTPS origin"}/twilio/sms · /twilio/voice · /twilio/voice/response (local listener port ${t.webhook_port||8767})`);
    }
    document.getElementById("twilio-form").addEventListener("input",()=>twilioDirty=true);
    document.getElementById("twilio-form").addEventListener("change",()=>twilioDirty=true);
    const changedProfileRoles=new Map();
    function renderFace(f){
      if(!f)return;put("face-status",f.status);put("face-identity",f.identity||"None");
      document.getElementById("face-count").textContent=String((f.profiles||[]).length);put("face-message",f.message);
      put("face-profile-info",f.profile_info||"");
      const root=document.getElementById("profiles");root.replaceChildren();
      for(const profile of (f.profiles||[])){
        const row=document.createElement("div");row.className="profile";
        const details=document.createElement("div");const name=document.createElement("strong");name.textContent=profile.name;
        const info=document.createElement("div");info.className="profile-info";info.textContent=`${profile.role||"user"} · ${profile.info||"No extra profile details."}`;
        details.append(name,info);const actions=document.createElement("div");actions.className="control-row";
        const role=document.createElement("select");role.className="control-button";
        const savedRole=profile.role||"user";
        const pendingRole=changedProfileRoles.get(profile.id);
        if(pendingRole===savedRole)changedProfileRoles.delete(profile.id);
        const selectedRole=changedProfileRoles.get(profile.id)||savedRole;
        for(const [value,label] of [["guest","Guest"],["user","User"],["admin","Admin"]]){const option=document.createElement("option");option.value=value;option.textContent=label;option.selected=selectedRole===value;role.append(option);}
        role.onchange=async()=>{
          const nextRole=role.value;changedProfileRoles.set(profile.id,nextRole);
          if(nextRole==="admin"&&!confirm(`Grant ${profile.name} Admin tool permissions? Elevated commands still require terminal confirmation and the OS sudo credential.`)){
            changedProfileRoles.delete(profile.id);role.value=profile.role||"user";return;
          }
          const saved=await postControl("/api/faces/permissions",{profile_id:profile.id,role:nextRole},"enroll-message");
          if(!saved){changedProfileRoles.delete(profile.id);role.value=profile.role||"user";}
        };
        const remove=document.createElement("button");remove.className="control-button danger";remove.textContent="Remove";
        remove.onclick=async()=>{if(!confirm(`Remove the local face profile for ${profile.name}?`))return;await postControl("/api/faces/delete",{profile_id:profile.id},"enroll-message");};
        actions.append(role,remove);row.append(details,actions);root.append(row);
      }
    }
    async function postControl(path,data,messageId){
      const target=document.getElementById(messageId);target.textContent="Saving…";
      try{const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)});const body=await r.json();if(!r.ok)throw new Error(body.error||"Request failed");target.textContent=body.message||"Done.";}
      catch(e){target.textContent=e.message;return false;}
      return true;
    }
    document.getElementById("settings-form").addEventListener("submit",async e=>{
      e.preventDefault();const f=e.currentTarget;const saved=await postControl("/api/settings",{
        mode:f.elements.mode.value,wake_word:f.elements.wake_word.value,
        tts_voice:f.elements.tts_voice.value,tts_speed:Number(f.elements.tts_speed.value),
        motion_cooldown:Number(f.elements.motion_cooldown.value),default_location:f.elements.default_location.value,
        whisper_model:f.elements.whisper_model.value,keep_alive:f.elements.keep_alive.value,
        max_agent_steps:Number(f.elements.max_agent_steps.value),max_history:Number(f.elements.max_history.value),
        max_facts:Number(f.elements.max_facts.value),motion_poll:Number(f.elements.motion_poll.value),
        command_silence_seconds:Number(f.elements.command_silence_seconds.value),
        command_max_seconds:Number(f.elements.command_max_seconds.value),
        presence_leave_seconds:Number(f.elements.presence_leave_seconds.value),
        wake_word_enabled:f.elements.wake_word_enabled.checked,motion_announce:f.elements.motion_announce.checked,
        presence_announce:f.elements.presence_announce.checked
      },"settings-message");if(saved)settingsDirty=false;
    });
    document.getElementById("twilio-form").addEventListener("submit",async e=>{
      e.preventDefault();const f=e.currentTarget;
      const saved=await postControl("/api/twilio/settings",{
        enabled:f.elements.enabled.checked,account_sid:f.elements.account_sid.value,
        auth_token:f.elements.auth_token.value,phone_number:f.elements.phone_number.value,
        webhook_url:f.elements.webhook_url.value,
        allowed_numbers:f.elements.allowed_numbers.value.split(/[ ,\\t\\r\\n]+/).filter(Boolean),
        inbound_sms:f.elements.inbound_sms.checked,inbound_voice:f.elements.inbound_voice.checked,
        outbound_sms:f.elements.outbound_sms.checked,outbound_voice:f.elements.outbound_voice.checked
      },"twilio-message");
      if(saved){twilioDirty=false;f.elements.auth_token.value="";}
    });
    document.getElementById("enroll-form").addEventListener("submit",async e=>{
      e.preventDefault();const f=e.currentTarget;if(f.elements.role.value==="admin"&&!confirm("Grant this profile Admin tool permissions? Elevated commands still require terminal confirmation and the OS sudo credential."))return;await postControl("/api/faces/enroll",{name:f.elements.name.value,info:f.elements.info.value,role:f.elements.role.value},"enroll-message");
    });
    document.getElementById("cancel-enrollment").onclick=()=>postControl("/api/faces/cancel",{},"enroll-message");
    document.getElementById("face-start").onclick=()=>postControl("/api/faces/start",{},"enroll-message");
    document.getElementById("face-stop").onclick=()=>postControl("/api/faces/stop",{},"enroll-message");
    async function refresh(){
      try{const response=await fetch("/api/state",{cache:"no-store"});if(!response.ok)throw new Error("offline");const s=await response.json();
        put("status",s.status);put("model",s.model);put("mode",s.mode);put("activity",s.activity);
        put("telegram-status",` · Telegram: ${s.telegram||"not-configured"}`);
        renderPerformance(s.performance);
        renderTools(s.tools);
        renderSettings(s.settings);renderFace(s.facial);
        renderTwilio(s.twilio);
        renderList("messages",s.messages,"message");renderList("events",s.events,"event");
      }catch(_){put("status","Disconnected");}
    }
    refresh();setInterval(refresh,1200);
  </script>
</body>
</html>""".encode("utf-8")


class _DashboardHandler(BaseHTTPRequestHandler):
    def _send_body(self, status, body, content_type):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            print("[dashboard] Client disconnected before the response was sent.")
        except OSError as exc:
            print(f"[dashboard] Response write failed: {type(exc).__name__}: {exc}")

    def _json_response(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_body(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = _PAGE
            content_type = "text/html; charset=utf-8"
        elif path == "/api/state":
            state = snapshot()
            state["performance"] = performance_snapshot()
            body = json.dumps(state, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif path == "/api/settings":
            body = json.dumps(snapshot().get("settings", {}), ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            self._json_response(404, {"error": "Not found."})
            return

        self._send_body(200, body, content_type)

    def do_POST(self):
        try:
            client = ipaddress.ip_address(self.client_address[0])
            if not client.is_loopback or self.headers.get("Host") != f"{HOST}:{PORT}":
                self._json_response(403, {"error": "Local dashboard requests only."})
                return
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{HOST}:{PORT}":
                self._json_response(403, {"error": "Cross-origin dashboard requests are blocked."})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 8192:
                self._json_response(400, {"error": "Invalid request size."})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            path = self.path.split("?", 1)[0]
            if path == "/api/settings" and _settings_handler:
                updated = _settings_handler(payload)
                self._json_response(200, {
                    "message": "Settings saved. Speech model changes require restart; startup mode remains Fast (4B).",
                    "settings": updated,
                })
            elif path == "/api/twilio/settings" and _twilio_service:
                updated = _twilio_service.update_settings(payload)
                set_state(twilio=updated)
                self._json_response(200, {"message": "Twilio settings saved locally.", "settings": updated})
            elif path == "/api/faces/enroll" and _face_service:
                _face_service.start_enrollment(payload.get("name"), payload.get("info", ""))
                self._json_response(200, {"message": "Enrollment started. Look at the camera and move your head slightly."})
            elif path == "/api/faces/cancel" and _face_service:
                _face_service.cancel_enrollment()
                self._json_response(200, {"message": "Enrollment cancelled."})
            elif path == "/api/faces/start" and _face_service:
                if not _face_service.start():
                    raise RuntimeError("Install opencv-contrib-python-headless to enable facial recognition.")
                self._json_response(200, {"message": "Local face recognition started."})
            elif path == "/api/faces/stop" and _face_service:
                _face_service.stop()
                self._json_response(200, {"message": "Local face recognition stopped."})
            elif path == "/api/faces/delete" and _face_service:
                message = _face_service.delete_profile(payload.get("profile_id", ""))
                self._json_response(200, {"message": message or "Face profile removed from local storage."})
            elif path == "/api/faces/permissions" and _face_service:
                _face_service.set_profile_role(payload.get("profile_id", ""), payload.get("role", ""))
                self._json_response(200, {"message": "Profile permissions updated."})
            else:
                self._json_response(404, {"error": "Control endpoint unavailable."})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json_response(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._json_response(409, {"error": str(exc)})
        except OSError as exc:
            print(f"[dashboard] Control operation failed: {type(exc).__name__}: {exc}")
            self._json_response(500, {"error": "The requested change could not be saved or completed."})
        except Exception as exc:
            print(f"[dashboard] Control operation failed: {type(exc).__name__}: {exc}")
            self._json_response(500, {"error": "The dashboard could not complete that request."})

    def log_message(self, fmt, *args):
        return


class _DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_dashboard(open_browser=True):
    """Start a local-only dashboard server and optionally open its browser UI."""
    global _server
    if _server is None:
        try:
            server = _DashboardServer((HOST, PORT), _DashboardHandler)
        except OSError as exc:
            raise RuntimeError(f"Could not start dashboard at {URL}: {exc}") from exc
        thread = threading.Thread(target=_serve_dashboard, args=(server,),
                                  name="jarvis-dashboard", daemon=True)
        _server = server
        try:
            thread.start()
        except Exception as exc:
            _server = None
            server.server_close()
            raise RuntimeError(f"Could not start dashboard server: {type(exc).__name__}: {exc}") from exc
    if open_browser:
        opened = False
        for name in ("firefox", "chromium", "chromium-browser", "google-chrome", "brave-browser"):
            browser = shutil.which(name)
            if not browser:
                continue
            try:
                subprocess.Popen(
                    [browser, URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                opened = True
                break
            except OSError as exc:
                print(f"[dashboard] Could not start {name}: {exc}")
        if not opened:
            try:
                opened = webbrowser.open(URL, new=2)
            except Exception as exc:
                print(f"[dashboard] Browser launch failed: {exc}")
        if not opened:
            print(f"[dashboard] Open {URL} in a browser.")
    return URL


def _serve_dashboard(server):
    global _server
    try:
        server.serve_forever()
    except Exception as exc:
        print(f"[dashboard] Server stopped unexpectedly: {type(exc).__name__}: {exc}")
    finally:
        try:
            server.server_close()
        except OSError as exc:
            print(f"[dashboard] Server cleanup failed: {type(exc).__name__}: {exc}")
        if _server is server:
            _server = None


def configure_controls(settings_handler=None, face_service=None, twilio_service=None):
    """Register local control handlers before serving the dashboard."""
    global _settings_handler, _face_service, _twilio_service
    _settings_handler = settings_handler
    _face_service = face_service
    _twilio_service = twilio_service
