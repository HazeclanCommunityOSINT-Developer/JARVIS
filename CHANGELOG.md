# Changelog

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
