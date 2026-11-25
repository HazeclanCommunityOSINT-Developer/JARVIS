"""Long-polling Telegram bot daemon for JARVIS.

Reads credentials from ~/.config/jarvis/telegram.json. This module uses only
the Python standard library and never exposes shell/admin tools to Telegram.
"""
import json
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class TelegramBotDaemon:
    def __init__(self, token, allowed_user_ids, on_message, on_status=None,
                 on_event=None):
        token = str(token or "").strip()
        if not token or ":" not in token:
            raise ValueError("Telegram bot token is missing or malformed.")
        self._token = token
        self._allowed_users = {int(user_id) for user_id in allowed_user_ids}
        self._on_message = on_message
        self._on_status = on_status or (lambda _status: None)
        self._on_event = on_event or (lambda _kind, _detail: None)
        self._stop = threading.Event()
        self._thread = None
        self._offset = None
        self._status = "configured"

    @classmethod
    def from_config(cls, config_path, on_message, on_status=None, on_event=None):
        try:
            if os.name == "posix" and os.path.exists(config_path):
                permissions = stat.S_IMODE(os.stat(config_path).st_mode)
                if permissions & 0o077:
                    raise RuntimeError("Telegram config is readable by other users; run chmod 600 on it.")
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read Telegram config: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError("Telegram config must contain a JSON object.")
        allowed_users = config.get("allowed_user_ids", [])
        if not isinstance(allowed_users, list):
            raise RuntimeError("allowed_user_ids must be a JSON array of numeric Telegram user IDs.")
        try:
            return cls(config.get("bot_token"), allowed_users, on_message,
                       on_status, on_event)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid Telegram config: {exc}") from exc

    @property
    def status(self):
        return self._status

    def _set_status(self, value):
        self._status = value
        self._on_status(value)

    def _add_event(self, detail):
        self._on_event("Telegram", detail)

    def _api(self, method, payload=None, timeout=35):
        url = f"https://api.telegram.org/bot{self._token}/{method}"
        fields = {key: value for key, value in (payload or {}).items() if value is not None}
        data = urllib.parse.urlencode(fields).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise RuntimeError(f"Telegram {method} request failed: {exc}") from exc
        if not result.get("ok"):
            raise RuntimeError(f"Telegram {method} failed with error code {result.get('error_code')}.")
        return result.get("result")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._set_status("starting")
        self._thread = threading.Thread(target=self._poll, name="jarvis-telegram", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._set_status("stopping")

    def _poll(self):
        backoff = 2
        try:
            webhook = self._api("getWebhookInfo", timeout=15) or {}
            if webhook.get("url"):
                self._set_status("webhook-conflict")
                self._add_event("A webhook is configured; long polling cannot receive messages.")
                print("[telegram] Webhook is configured; disable it to use long polling.")
                return
            self._set_status("online")
        except Exception as exc:
            self._set_status("connection-error")
            self._add_event(f"Telegram connection check failed: {type(exc).__name__}.")
        while not self._stop.is_set():
            try:
                updates = self._api("getUpdates", {
                    "offset": self._offset,
                    "timeout": 25,
                    "allowed_updates": json.dumps(["message"]),
                }) or []
                backoff = 2
                self._set_status("online")
                for update in updates:
                    self._offset = int(update.get("update_id", 0)) + 1
                    self._handle_update(update)
                    if self._stop.is_set():
                        break
            except Exception as exc:
                if not self._stop.is_set():
                    # Avoid printing request URLs because they contain the bot token.
                    print(f"[telegram] Poll failed: {type(exc).__name__}; retrying in {backoff}s")
                    self._add_event(f"Telegram polling failed: {type(exc).__name__}; retrying.")
                    self._set_status("connection-error")
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 60)
        self._set_status("stopped")

    def _handle_update(self, update):
        message = update.get("message") or {}
        sender = message.get("from") or {}
        chat = message.get("chat") or {}
        text = message.get("text")
        if not isinstance(text, str) or chat.get("type") != "private":
            return
        user_id = sender.get("id")
        chat_id = chat.get("id")
        if not isinstance(user_id, int) or not isinstance(chat_id, int):
            return
        self._add_event("Received a private message.")
        command = text.strip().lower().split("@", 1)[0]
        if command == "/id":
            self._send_message(chat_id, f"Your Telegram user ID is {user_id}.")
            self._add_event("Sent the /id response.")
            return
        # An empty allowlist means any private Telegram user may use the bot.
        # A populated list restricts access to the listed numeric user IDs.
        if self._allowed_users and user_id not in self._allowed_users:
            return
        normalized = text.strip()
        if normalized.lower() in {"/start", "/help"}:
            self._send_message(chat_id, "JARVIS Telegram is connected. Send a message to chat. Local shell, desktop controls, and administrator commands are disabled here.")
            self._add_event("Sent the Telegram help response.")
            return
        try:
            reply = self._on_message(user_id, normalized)
        except Exception as exc:
            print(f"[telegram] Message handling failed: {type(exc).__name__}")
            self._add_event(f"Message handling failed: {type(exc).__name__}.")
            reply = "I couldn't process that just now."
        self._send_message(chat_id, reply or "I've got nothing.")
        self._add_event("Sent a reply.")

    def _send_message(self, chat_id, text):
        chunks = _split_message(str(text), 4000)
        for chunk in chunks:
            self._api("sendMessage", {"chat_id": chat_id, "text": chunk}, timeout=20)


def _split_message(text, limit):
    text = text.strip()
    if not text:
        return [" "]
    chunks = []
    while len(text) > limit:
        split = text.rfind("\n", 0, limit)
        if split < limit // 2:
            split = text.rfind(" ", 0, limit)
        if split < limit // 2:
            split = limit
        chunks.append(text[:split].strip())
        text = text[split:].strip()
    if text:
        chunks.append(text)
    return chunks
