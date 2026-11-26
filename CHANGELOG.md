# Changelog

## v1.1.02 — 2026-11-25

### Added

- Reorganized the local dashboard into menu pages for overview, conversation,
  activity, camera, face recognition, and settings. Navigation follows the URL
  fragment and adapts to narrow screens.
- Telegram private chats can now use voice notes as spoken requests. JARVIS
  transcribes notes up to 2 minutes and 20 MB locally, then replies with text.
- Voice requests display a `[thinking...]` status in the console and dashboard
  while JARVIS routes or generates a response.
- After a spoken reply, JARVIS listens for a follow-up for 10 seconds, so you can
  continue the conversation without repeating the wake phrase.
- Presence greetings include the active model, speech model, wake settings, VLC
  availability, and downloaded-audio index count, delivered as a dry systems report.
- JARVIS indexes supported audio files under the system Downloads folder and can
  search for and play matching tracks with VLC.

### Fixed

- Added deadlines to microphone reads and shared STT/Ollama lock waits, plus a
  five-minute Ollama request timeout, so stalled capture or concurrent requests
  surface as errors instead of blocking the voice loop indefinitely.
- Stop face-recognition and motion-watcher threads before JARVIS exits, avoiding
  abrupt interpreter shutdown while native camera code is running.
- Defer the first Ctrl+C during native speech transcription for a clean shutdown;
  a second Ctrl+C forces exit if the native call does not return.
- Hardened settings, memory, and conversation-history loading against malformed
  or inaccessible files, and made JSON writes safer and atomic.
- Improved Ollama fallback and failure reporting for weather, web search, system
  commands, and speech output. Speech recording now shuts down more reliably,
  and text-to-speech cleans up temporary audio and falls back when playback fails.
- Made Telegram polling and message handling more resilient to invalid updates,
  API failures, and callback errors; diagnostics redact the bot token.
- Improved dashboard response handling and server recovery, plus face profile
  validation, enrollment cleanup, model rebuilding, and recognition recovery.

## v1.1 — 2026-11-25

### Added

- Added a local dashboard for runtime status, conversations, tool activity,
  camera streaming, and runtime settings.
- Added optional facial recognition with profile enrollment and Guest, User,
  and Admin tool permissions. Admin commands require typed confirmation and
  fresh system authentication.
- Added an optional Telegram bot using Bot API long polling. Telegram access is
  limited to private chats and normal conversation, time, weather, and web
  search; shell, desktop controls, personal memory, and Admin commands are not
  exposed.

### Changed

- An empty Telegram `allowed_user_ids` list now allows private users to chat;
  a populated list restricts access to those IDs.
- Telegram configuration is read from `~/.config/jarvis/telegram.json` and
  kept outside the project repository.
- Added Telegram connection status, received-message activity, and webhook
  conflict diagnostics to the dashboard and terminal.
- Bundled the Python runtime dependencies for the supported Linux/Python 3.13
  setup.
