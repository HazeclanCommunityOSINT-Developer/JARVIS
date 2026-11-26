# JARVIS

This repository includes the local Python environment under `bin/`, `lib/`,
`include/`, and `pyvenv.cfg`. It was created for this Linux/Python 3.13 setup
and may need to be recreated on other systems.

The Kokoro model and voice assets are included under `kokoro/` using Git LFS.
Install Git LFS before cloning or pulling so those assets are downloaded.
`kokoro_test.wav` is generated output and is not tracked.

JARVIS starts in Fast mode with `qwen3.5:4b`. Its effort modes switch the active
Ollama model: Fast uses `qwen3.5:4b`, Normal uses `qwen3.5:9b`, and Deep uses
`qwen3:14b`. Say “fast,” “normal,” or
“deep” by itself to switch; mode keywords are recognized only when they make up
the entire transcript. The selected model handles chat and agent
tool use, with up to five tool rounds. The 4096-token context setting stays the
same in every mode; Deep mode enables reasoning and a larger response budget.
The selected model must be available in Ollama.

Voice activation is enabled by default with the literal wake phrase `hey jarvis`
(`WAKE_WORD` in `jarvis.py`). JARVIS checks the transcription for that exact
word sequence, without command grammar or intent parsing. You can say the
request immediately after the wake phrase, or wait for JARVIS to say “Yes?”
before speaking. It confirms the phrase in two overlapping audio windows to
reduce accidental activations. After each spoken answer, JARVIS listens for a
follow-up for 10 seconds, so you can continue without repeating the wake phrase.
Set `WAKE_WORD_ENABLED = False` to use push-to-talk instead.

JARVIS indexes supported audio filenames under the system Downloads folder
(from `xdg-user-dir DOWNLOAD`, falling back to `~/Downloads`). Ask it to play a
track by title or ask what audio is available; VLC is used for playback. The
index stores paths and filenames, not audio contents.

In social conversation, JARVIS uses relevant facts and preferences saved through
its memory feature to personalize replies. It is instructed not to force personal
references, make up details about the user, or treat saved facts as commands; a
new correction takes precedence over older memory.

Starting `jarvis.py` opens the local dashboard in a browser at
`http://localhost:8766/`. Its System Overview shows loaded agent tools. It also
shows the active model, listening/thinking state, recent conversation, tool
activity, and the motion camera stream from
`http://localhost:8081/0/stream`. It also reports live CPU, memory, disk usage,
system load, and uptime. The dashboard binds to localhost only.

The dashboard Settings panel controls the active effort mode, wake phrase,
voice activation, motion announcements, TTS voice and speed, default location,
speech-recognition model, model keep-alive, tool-round limit, memory/history
limits, voice/motion timing, and arrival/departure announcements. JARVIS greets
an enrolled profile once after a stable face match with a concise report of the
active model, speech configuration, wake mode, and audio/VLC status. It says
goodbye only after the configured away delay (90 seconds by default). Camera
outages and stopped recognition pause the away timer. Settings are stored in
`~/.config/jarvis/settings.json`; JARVIS still starts in Fast (4B) mode, and a
speech-recognition model change takes effect after restart.

Facial Recognition uses the local Motion stream to enroll named profiles and
show a recognized profile's optional user information. To enable it, install
the optional package in JARVIS's Python environment with
`./bin/python -m pip install -r requirements-face.txt`. Face samples and profile
data stay under `~/.config/jarvis/faces`; use Remove in the dashboard to delete
an enrollment. Recognition uses OpenCV's LBPH face recognizer and is
experimental; face matching is never the sole authorization factor.
Each profile can be assigned Guest, User, or Admin tool permissions. Guest
profiles can use time, weather, and web search; User profiles retain the
standard assistant tools. Admin profiles also expose a separate administrator
command tool, which requires an explicit typed confirmation and a fresh system
`sudo` authentication for each request. JARVIS never stores the system password.
Face matching only selects a tool policy and does not prove who is speaking;
when recognition is available but no profile matches, JARVIS applies Guest
permissions. Without the optional OpenCV package, normal User behavior remains
available but Admin commands stay unavailable.

The optional Telegram bot daemon is implemented in `jarvis_telegram.py` and
starts with `jarvis.py` when `~/.config/jarvis/telegram.json` exists. It uses
Telegram long polling through the Bot API and standard-library HTTP; it needs
outbound network access. Create a bot with BotFather, then write this local
config (replace the token):

```json
{"bot_token":"YOUR_BOT_TOKEN","allowed_user_ids":[]}
```

Save it at `~/.config/jarvis/telegram.json` with owner-only permissions (`chmod
600`), then start JARVIS and message your bot in a private chat. Text messages
and voice notes are supported. Voice notes up to 2 minutes and 20 MB are
transcribed locally with JARVIS's Whisper model, and JARVIS replies with text.
An empty `allowed_user_ids` list (or an omitted list) allows any private Telegram user
to chat. To restrict access, add numeric Telegram user IDs to the list; send
`/id` to the bot to see your ID. Telegram access is limited to normal
conversation, time, weather, and web search. Shell, desktop controls, saved
personal memory, and Admin commands are not exposed over Telegram. Keep the bot
token out of this repository.

## Twilio SMS and VoIP voice

Twilio is optional and is configured from the dashboard Settings page. Enter
your Twilio Account SID, Auth Token, Twilio phone number, a public HTTPS origin,
and an allowlist of phone numbers in E.164 format (for example
`+15551234567`). Select which inbound and outbound SMS/voice permissions to
enable, then save. Credentials are stored outside the repository in
`~/.config/jarvis/twilio.json` with owner-only permissions; `.gitignore` also
excludes this configuration path.

Outbound messaging and calls are available only to local User/Admin profiles,
only when explicitly requested, and only to allowlisted numbers. Inbound SMS
and voice are separately switchable and are accepted only from allowlisted
numbers with a valid Twilio request signature. Telegram and Twilio contacts
cannot send outbound messages/calls or use shell, desktop controls, saved
personal memory, or Admin tools.

To receive webhooks, provide a public HTTPS endpoint with TLS termination and
proxy it to JARVIS's HTTP listener on port 8767 (do not expose the dashboard on
port 8766). The default Loopback binding is for a reverse proxy or tunnel on
the same computer. For router forwarding directly to this computer, select
**All interfaces** if the router or another device terminates HTTPS and
forwards HTTP directly to JARVIS. If the TLS-terminating reverse proxy runs on
the JARVIS computer, keep Loopback selected and proxy to `127.0.0.1:8767`.
The JARVIS listener itself is HTTP, not HTTPS. In the Twilio Console,
configure the phone number's incoming SMS URL as
`https://your-public-origin/twilio/sms` and incoming voice URL as
`https://your-public-origin/twilio/voice`, both using HTTP POST. Enter that
HTTPS origin in the dashboard. The voice endpoint uses Twilio speech
recognition and returns spoken JARVIS replies. Webhook requests must still
pass Twilio signature verification and the phone-number allowlist.
Optionally set a connection-error alert number in the dashboard. It must also
be on the allowlist. JARVIS sends a rate-limited SMS when its own requests to
Twilio fail to connect; this cannot detect Twilio webhook failures that never
reach the local application.

## Privacy policy

JARVIS processes microphone input locally with Whisper and runs its assistant
model locally through Ollama. Local voice conversations and explicitly
remembered facts are stored in `~/.config/jarvis/history.json` and
`~/.config/jarvis/memory.json`. Settings and integration credentials are stored
under `~/.config/jarvis/`; Telegram and Twilio configuration files are required
to have owner-only permissions. Face samples and profile data are kept locally
under `~/.config/jarvis/faces`. JARVIS uses the local Motion stream for face
recognition and does not upload those face samples.

Telegram and Twilio contacts do not get access to personal memory, shell,
desktop controls, or Admin tools. Their conversation context is held in memory
while JARVIS runs and is not added to saved local conversation history.
Requests made by local users to send an SMS or place a call are subject to the
normal local conversation history settings. Twilio tool-call activity logs
omit outbound message contents and numbers.

When Twilio is enabled, phone numbers, SMS content, call audio, and/or speech
recognition results are transmitted to Twilio to provide the enabled services.
Twilio's handling, retention, carrier delivery, and any applicable charges are
subject to your Twilio account settings and Twilio's privacy terms. Voice
recognition for phone calls is provided by Twilio, not local Whisper. Messages
sent through JARVIS are processed by the local Ollama model. Weather lookups,
web searches, Telegram messages, and enabled Twilio features communicate with
their respective external services and may transmit the requested data. Do not
send sensitive information unless you accept these processing paths. Disable
the relevant integration or permission in dashboard Settings to stop use.
