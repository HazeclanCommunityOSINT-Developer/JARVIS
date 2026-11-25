"""Local read-only web dashboard for JARVIS runtime state."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
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
_state = {
    "status": "starting",
    "activity": "Starting JARVIS",
    "model": "qwen3.5:4b",
    "mode": "FAST",
    "tools": [],
    "telegram": "not-configured",
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


_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'self' http://localhost:8081">
  <title>JARVIS Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1018; --panel:#121b28; --line:#243247; --ink:#e8f0fa; --muted:#8da2bb; --cyan:#64d5e7; --gold:#f2c879; }
    * { box-sizing:border-box }
    body { margin:0; background:radial-gradient(ellipse at top,#17273b,var(--bg) 58%); color:var(--ink); font:15px/1.5 system-ui,sans-serif; min-height:100vh }
    main { width:min(1120px,94vw); margin:36px auto; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:24px }
    h1 { font-size:clamp(27px,5vw,42px); letter-spacing:.12em; margin:0 }
    .subtitle,.muted { color:var(--muted) }
    .badge { border:1px solid var(--line); border-radius:999px; padding:7px 13px; color:var(--cyan); background:#0e1824; text-transform:uppercase; font-size:12px; letter-spacing:.1em }
    .grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:18px }
    .card,.panel { background:linear-gradient(145deg,#162333,var(--panel)); border:1px solid var(--line); border-radius:16px; box-shadow:0 12px 30px #0004 }
    .overview { margin-bottom:18px; min-height:0 }
    .overview-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px }
    .overview-head h2 { margin:0 }
    .tool-count { color:var(--muted); font-size:12px }
    .tools { display:flex; flex-wrap:wrap; gap:8px }
    .tool { border:1px solid #2c4359; border-radius:999px; background:#0e1824; color:var(--cyan); padding:5px 10px; font-size:12px }
    .card { padding:18px 20px; min-height:105px }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.12em }
    .value { margin-top:8px; font-size:20px; font-weight:650; overflow-wrap:anywhere }
    .columns { display:grid; grid-template-columns:1.3fr .7fr; gap:14px }
    .panel { padding:20px; min-height:300px }
    .video-panel { margin-bottom:14px; min-height:0 }
    .control-panel { margin-bottom:14px; min-height:0 }
    .settings-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px }
    .setting { display:grid; gap:5px; color:var(--muted); font-size:12px }
    .setting input,.setting select { width:100%; border:1px solid var(--line); border-radius:8px; padding:9px; background:#0b141f; color:var(--ink); font:inherit }
    .setting.check { display:flex; align-items:center; gap:8px; padding-top:22px }
    .setting.check input { width:auto }
    .control-button { border:1px solid #3c627d; border-radius:8px; padding:9px 13px; color:var(--ink); background:#173048; cursor:pointer }
    .control-button.danger { border-color:#74434b; background:#351c24 }
    .control-row { display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-top:14px }
    .control-message { color:var(--muted); font-size:12px }
    .face-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:12px }
    .face-value { border:1px solid var(--line); background:#0e1722; padding:10px; border-radius:9px; overflow-wrap:anywhere }
    .profiles { display:grid; gap:8px; margin-top:12px }
    .profile { display:flex; justify-content:space-between; gap:12px; align-items:center; padding:9px 11px; background:#0e1722; border-radius:9px }
    .profile-info { color:var(--muted); font-size:12px; white-space:pre-wrap }
    .stream-frame { width:100%; aspect-ratio:16/9; display:grid; place-items:center; overflow:hidden; border-radius:10px; background:#080d14; border:1px solid var(--line) }
    .stream-frame img { display:block; width:100%; height:100%; object-fit:contain }
    .stream-note { color:var(--muted); font-size:12px; margin-top:8px }
    h2 { font-size:14px; text-transform:uppercase; letter-spacing:.13em; margin:0 0 14px; color:var(--gold) }
    .list { display:grid; gap:10px; max-height:480px; overflow:auto }
    .entry { border-left:2px solid var(--cyan); padding:8px 11px; background:#0e1722; border-radius:0 9px 9px 0; overflow-wrap:anywhere }
    .entry.assistant { border-color:var(--gold) }
    .meta { color:var(--muted); font-size:11px; margin-bottom:3px; text-transform:uppercase; letter-spacing:.08em }
    .empty { color:var(--muted); padding:10px 0 }
    footer { color:var(--muted); font-size:12px; text-align:right; margin-top:14px }
    @media(max-width:720px) { .grid,.columns,.face-summary,.settings-grid { grid-template-columns:1fr } header { align-items:flex-start; flex-direction:column } }
  </style>
</head>
<body>
  <main>
    <header><div><h1>JARVIS</h1><div class="subtitle">Local assistant dashboard</div></div><div id="status" class="badge">Starting</div></header>
    <section class="panel overview">
      <div class="overview-head"><h2>System Overview</h2><div><span id="tool-count" class="tool-count">Loading tools…</span><span id="telegram-status" class="tool-count"> · Telegram: not configured</span></div></div>
      <div id="tools" class="tools"><span class="muted">Waiting for JARVIS startup…</span></div>
    </section>
    <section class="panel control-panel">
      <h2>Facial Recognition · Local</h2>
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
    </section>
    <section class="panel control-panel">
      <h2>Dashboard Settings</h2>
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
    </section>
    <section class="grid">
      <article class="card"><div class="label">Active model</div><div id="model" class="value">qwen3.5:4b</div></article>
      <article class="card"><div class="label">Effort mode</div><div id="mode" class="value">FAST</div></article>
      <article class="card"><div class="label">Current activity</div><div id="activity" class="value">Starting JARVIS</div></article>
    </section>
    <section class="panel video-panel">
      <h2>Motion camera</h2>
      <div class="stream-frame"><img src="http://localhost:8081/0/stream" alt="Live motion camera stream" onerror="document.getElementById('stream-status').textContent='Stream unavailable at localhost:8081/0/stream'"></div>
      <div id="stream-status" class="stream-note">Live stream: localhost:8081/0/stream</div>
    </section>
    <section class="columns">
      <article class="panel"><h2>Conversation</h2><div id="messages" class="list"><div class="empty">Waiting for conversation…</div></div></article>
      <article class="panel"><h2>Recent activity</h2><div id="events" class="list"><div class="empty">No activity yet.</div></div></article>
    </section>
    <footer>Served locally at localhost:8766 · refreshes automatically</footer>
  </main>
  <script>
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
    function renderSettings(s){
      if(!s||settingsDirty)return;const form=document.getElementById("settings-form");
      for(const key of ["mode","wake_word","tts_voice","tts_speed","motion_cooldown","default_location","whisper_model","keep_alive","max_agent_steps","max_history","max_facts","motion_poll","command_silence_seconds","command_max_seconds","presence_leave_seconds"]){if(s[key]!==undefined)form.elements[key].value=s[key];}
      for(const key of ["wake_word_enabled","motion_announce","presence_announce"]){if(s[key]!==undefined)form.elements[key].checked=!!s[key];}
    }
    let settingsDirty=false;
    document.getElementById("settings-form").addEventListener("input",()=>settingsDirty=true);
    document.getElementById("settings-form").addEventListener("change",()=>settingsDirty=true);
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
        const selectedRole=changedProfileRoles.get(profile.id)||(profile.role||"user");
        for(const [value,label] of [["guest","Guest"],["user","User"],["admin","Admin"]]){const option=document.createElement("option");option.value=value;option.textContent=label;option.selected=selectedRole===value;role.append(option);}
        role.onchange=async()=>{
          const nextRole=role.value;changedProfileRoles.set(profile.id,nextRole);
          if(nextRole==="admin"&&!confirm(`Grant ${profile.name} Admin tool permissions? Elevated commands still require terminal confirmation and the OS sudo credential.`)){
            changedProfileRoles.delete(profile.id);role.value=profile.role||"user";return;
          }
          if(await postControl("/api/faces/permissions",{profile_id:profile.id,role:nextRole},"enroll-message"))changedProfileRoles.delete(profile.id);
          else role.value=profile.role||"user";
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
        renderTools(s.tools);
        renderSettings(s.settings);renderFace(s.facial);
        renderList("messages",s.messages,"message");renderList("events",s.events,"event");
      }catch(_){put("status","Disconnected");}
    }
    refresh();setInterval(refresh,1200);
  </script>
</body>
</html>""".encode("utf-8")


class _DashboardHandler(BaseHTTPRequestHandler):
    def _json_response(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = _PAGE
            content_type = "text/html; charset=utf-8"
        elif path == "/api/state":
            body = json.dumps(snapshot(), ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        elif path == "/api/settings":
            body = json.dumps(snapshot().get("settings", {}), ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

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
                _face_service.delete_profile(payload.get("profile_id", ""))
                self._json_response(200, {"message": "Face profile removed from local storage."})
            elif path == "/api/faces/permissions" and _face_service:
                _face_service.set_profile_role(payload.get("profile_id", ""), payload.get("role", ""))
                self._json_response(200, {"message": "Profile permissions updated."})
            else:
                self._json_response(404, {"error": "Control endpoint unavailable."})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json_response(400, {"error": str(exc)})
        except Exception as exc:
            self._json_response(409, {"error": str(exc)})

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
            _server = _DashboardServer((HOST, PORT), _DashboardHandler)
        except OSError as exc:
            raise RuntimeError(f"Could not start dashboard at {URL}: {exc}") from exc
        threading.Thread(target=_server.serve_forever, name="jarvis-dashboard", daemon=True).start()
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


def configure_controls(settings_handler=None, face_service=None):
    """Register local control handlers before serving the dashboard."""
    global _settings_handler, _face_service
    _settings_handler = settings_handler
    _face_service = face_service
