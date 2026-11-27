"""Optional Twilio SMS and voice bridge using the Python standard library."""
import base64
import hashlib
import hmac
import json
import os
import re
import stat
import threading
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.sax.saxutils import escape


PHONE_RE = re.compile(r"^\+[1-9]\d{7,14}$")
MAX_BODY_BYTES = 16 * 1024
WEBHOOK_BIND_HOSTS = {
    "loopback": "172.168.0.155",
    "all_interfaces": "172.168.0.155",
}
WEBHOOK_PORT = 8767


def _phone(value):
    return isinstance(value, str) and PHONE_RE.fullmatch(value) is not None


class TwilioBridge:
    def __init__(self, config_path, on_message, on_event=None):
        self.config_path = config_path
        self.on_message = on_message
        self.on_event = on_event or (lambda _kind, _detail: None)
        self._lock = threading.RLock()
        self._config = self._load_config()
        self._server = None
        self._thread = None
        self._voice_histories = {}
        self._message_lock = threading.Lock()
        self._seen_messages = set()
        self._seen_message_order = []

    def _load_config(self):
        try:
            if os.name == "posix" and os.path.exists(self.config_path):
                permissions = stat.S_IMODE(os.stat(self.config_path).st_mode)
                if permissions & 0o077:
                    raise RuntimeError("Twilio config is readable by other users; run chmod 600 on it.")
            with open(self.config_path, encoding="utf-8") as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            return self._defaults()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read Twilio config: {exc}") from exc
        if not isinstance(config, dict):
            raise RuntimeError("Twilio config must contain a JSON object.")
        try:
            return self._validate(config)
        except ValueError as exc:
            raise RuntimeError(f"Invalid Twilio config: {exc}") from exc

    @staticmethod
    def _defaults():
        return {
            "enabled": True, "account_sid": "", "auth_token": "",
            "phone_number": "", "webhook_url": "", "allowed_numbers": [],
            "webhook_bind": "loopback",
            "connection_alert_number": "",
            "inbound_sms": True, "inbound_voice": True,
            "outbound_sms": True, "outbound_voice": True,
        }

    @staticmethod
    def _validate(config):
        validated = TwilioBridge._defaults()
        for key in ("enabled", "inbound_sms", "inbound_voice",
                    "outbound_sms", "outbound_voice"):
            value = config.get(key, False)
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false.")
            validated[key] = value
        for key in ("account_sid", "auth_token", "phone_number", "webhook_url",
                    "connection_alert_number"):
            value = config.get(key, "")
            if not isinstance(value, str):
                raise ValueError(f"{key} must be text.")
            validated[key] = value.strip()
        webhook_bind = config.get("webhook_bind", "loopback")
        if webhook_bind not in WEBHOOK_BIND_HOSTS:
            raise ValueError("Webhook bind must be loopback or all_interfaces.")
        validated["webhook_bind"] = webhook_bind
        if validated["account_sid"] and not re.fullmatch(r"AC[0-9a-fA-F]{32}", validated["account_sid"]):
            raise ValueError("Twilio Account SID must start with AC and contain 32 hexadecimal characters.")
        if validated["phone_number"] and not _phone(validated["phone_number"]):
            raise ValueError("Twilio phone number must use E.164 format, such as +15551234567.")
        if validated["webhook_url"]:
            parsed = urllib.parse.urlsplit(validated["webhook_url"])
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError("Public webhook URL has an invalid port.") from exc
            valid_host = (parsed.hostname and all(
                char.isalnum() or char in ".-:[]" for char in parsed.netloc))
            if (parsed.scheme != "https" or not parsed.netloc or not valid_host or parsed.path
                    or parsed.query or parsed.fragment or parsed.username or parsed.password):
                raise ValueError("Public webhook URL must be an HTTPS origin with no path, query, or credentials.")
        numbers = config.get("allowed_numbers", [])
        if not isinstance(numbers, list) or any(not _phone(number) for number in numbers):
            raise ValueError("Allowed numbers must be a list of E.164 phone numbers.")
        validated["allowed_numbers"] = sorted(set(numbers))
        if (validated["connection_alert_number"]
                and (not _phone(validated["connection_alert_number"])
                     or validated["connection_alert_number"] not in validated["allowed_numbers"])):
            raise ValueError("Connection alert number must be an allowlisted E.164 phone number.")
        if validated["enabled"]:
            if not validated["account_sid"] or not validated["auth_token"]:
                raise ValueError("Enter both a Twilio Account SID and Auth Token to enable Twilio.")
            if not validated["phone_number"]:
                raise ValueError("Enter the Twilio phone number in E.164 format.")
            if not validated["allowed_numbers"]:
                raise ValueError("Add at least one allowed phone number before enabling Twilio.")
            if (validated["inbound_sms"] or validated["inbound_voice"]) and not validated["webhook_url"]:
                raise ValueError("Set the public HTTPS webhook URL to enable inbound SMS or voice.")
        return validated

    def settings(self):
        with self._lock:
            config = dict(self._config)
        config["auth_token_configured"] = bool(config.get("auth_token"))
        config["auth_token"] = ""
        config["webhook_port"] = WEBHOOK_PORT
        return config

    def update_settings(self, values):
        if not isinstance(values, dict):
            raise ValueError("Twilio settings must be an object.")
        allowed = {"enabled", "account_sid", "auth_token", "phone_number",
                   "webhook_url", "allowed_numbers", "inbound_sms",
                   "inbound_voice", "outbound_sms", "outbound_voice",
                   "connection_alert_number", "webhook_bind"}
        if set(values) != allowed:
            raise ValueError("Twilio settings request must include every supported setting.")
        candidate = dict(values)
        if not candidate["auth_token"]:
            candidate["auth_token"] = self._config.get("auth_token", "")
        candidate = self._validate(candidate)
        directory = os.path.dirname(self.config_path) or "."
        os.makedirs(directory, mode=0o700, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".twilio.", suffix=".tmp", dir=directory)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as config_file:
                json.dump(candidate, config_file, indent=2)
                config_file.flush()
                os.fsync(config_file.fileno())
            os.replace(temporary, self.config_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        with self._lock:
            self._config = candidate
        self._set_webhook_running(
            candidate["enabled"] and (candidate["inbound_sms"] or candidate["inbound_voice"]),
            candidate["webhook_bind"])
        return self.settings()

    def start(self):
        with self._lock:
            should_listen = (self._config["enabled"]
                             and (self._config["inbound_sms"] or self._config["inbound_voice"]))
            bind_host = self._config["webhook_bind"]
        self._set_webhook_running(should_listen, bind_host)

    def stop(self):
        with self._lock:
            server = self._server
            self._server = None
            self._thread = None
        if server:
            server.shutdown()
            server.server_close()

    def _set_webhook_running(self, enabled, bind_host="loopback"):
        address = WEBHOOK_BIND_HOSTS[bind_host]
        with self._lock:
            current = self._server
        if enabled and current and current.server_address[0] == address:
            return
        if current:
            with self._lock:
                self._server = None
                self._thread = None
            current.shutdown()
            current.server_close()
        if not enabled:
            return
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                bridge._handle_webhook(self)

            def log_message(self, _format, *_args):
                return

        try:
            server = ThreadingHTTPServer((address, WEBHOOK_PORT), Handler)
        except OSError as exc:
            raise RuntimeError(
                f"Could not start Twilio webhook listener on {address}:{WEBHOOK_PORT}: {exc}") from exc
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever,
                                  name="jarvis-twilio-webhook", daemon=True)
        with self._lock:
            self._server = server
            self._thread = thread
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._server = None
                self._thread = None
            server.server_close()
            raise

    def _event(self, detail):
        try:
            self.on_event("Twilio", detail)
        except Exception as exc:
            print(f"[twilio] Event callback failed: {type(exc).__name__}")

    @staticmethod
    def _respond(handler, status, content_type, body):
        encoded = body.encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(encoded)))
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        handler.wfile.write(encoded)

    def _handle_webhook(self, handler):
        with self._lock:
            config = dict(self._config)
        endpoint = urllib.parse.urlsplit(handler.path).path
        if endpoint not in {"/twilio/sms", "/twilio/voice", "/twilio/voice/response"}:
            self._respond(handler, 404, "text/plain; charset=utf-8", "Not found.")
            return
        if handler.headers.get_content_type() != "application/x-www-form-urlencoded":
            self._respond(handler, 415, "text/plain; charset=utf-8", "Unsupported media type.")
            return
        if handler.headers.get("Transfer-Encoding"):
            self._respond(handler, 400, "text/plain; charset=utf-8", "Invalid request.")
            return
        try:
            length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length < 1 or length > MAX_BODY_BYTES:
            self._respond(handler, 400, "text/plain; charset=utf-8", "Invalid request.")
            return
        raw_body = handler.rfile.read(length)
        try:
            fields = urllib.parse.parse_qs(raw_body.decode("utf-8"), keep_blank_values=True,
                                           strict_parsing=True)
        except (UnicodeError, ValueError):
            self._respond(handler, 400, "text/plain; charset=utf-8", "Invalid request.")
            return
        if not self._valid_signature(config, handler.path, fields,
                                     handler.headers.get("X-Twilio-Signature", "")):
            self._respond(handler, 403, "text/plain; charset=utf-8", "Forbidden.")
            return
        sender = fields.get("From", [""])[0]
        if not config["enabled"] or sender not in config["allowed_numbers"]:
            self._respond(handler, 403, "text/plain; charset=utf-8", "Forbidden.")
            return
        try:
            if endpoint == "/twilio/sms" and config["inbound_sms"]:
                text = fields.get("Body", [""])[0].strip()
                if text:
                    message_id = fields.get("MessageSid", [""])[0]
                    if message_id and self._claim_message(message_id):
                        thread = threading.Thread(
                            target=self._process_sms, args=(sender, text),
                            name="jarvis-twilio-sms", daemon=True)
                        thread.start()
                    elif message_id:
                        self._event("Ignored a duplicate inbound SMS webhook.")
                self._respond(handler, 200, "text/xml; charset=utf-8", "<Response/>")
            elif endpoint == "/twilio/voice" and config["inbound_voice"]:
                self._event("Received an allowlisted inbound voice call.")
                twiml = self._voice_prompt()
                self._respond(handler, 200, "text/xml; charset=utf-8", twiml)
            elif endpoint == "/twilio/voice/response" and config["inbound_voice"]:
                speech = fields.get("SpeechResult", [""])[0].strip()
                if not speech:
                    self._respond(handler, 200, "text/xml; charset=utf-8",
                                   self._voice_prompt("I didn't catch that."))
                    return
                call_id = fields.get("CallSid", [""])[0]
                history = self._voice_histories.setdefault(call_id, [])
                reply = self.on_message(f"{sender}:voice", speech, history)
                self._voice_histories[call_id] = history[-8:]
                if len(self._voice_histories) > 100:
                    self._voice_histories.pop(next(iter(self._voice_histories)))
                self._respond(handler, 200, "text/xml; charset=utf-8",
                              self._voice_prompt(reply))
                if fields.get("Confidence", [""])[0]:
                    self._event("Processed an allowlisted inbound voice turn.")
            else:
                self._respond(handler, 403, "text/plain; charset=utf-8", "Disabled.")
        except Exception as exc:
            print(f"[twilio] Webhook handling failed: {type(exc).__name__}: {exc}")
            self._event("Webhook processing failed.")
            self._respond(handler, 200, "text/xml; charset=utf-8",
                          "<Response><Say>I'm unable to help right now.</Say><Hangup/></Response>")

    def _claim_message(self, message_id):
        with self._message_lock:
            if message_id in self._seen_messages:
                return False
            self._seen_messages.add(message_id)
            self._seen_message_order.append(message_id)
            if len(self._seen_message_order) > 500:
                expired = self._seen_message_order.pop(0)
                self._seen_messages.discard(expired)
            return True

    def _process_sms(self, sender, text):
        self._event("Received an allowlisted inbound SMS.")
        try:
            reply = self.on_message(sender, text)
            self._send_sms(sender, reply, permission="inbound_sms")
        except Exception as exc:
            print(f"[twilio] Inbound SMS processing failed: {type(exc).__name__}: {exc}")
            self._event("Inbound SMS processing failed.")

    def _voice_prompt(self, message="Hello. This is JARVIS. What can I help you with?"):
        action = _xml_text(message)
        if len(action) > 1200:
            action = action[:1200]
        with self._lock:
            webhook_url = self._config["webhook_url"].rstrip("/")
        callback = escape(f"{webhook_url}/twilio/voice/response",
                          {'"': "&quot;", "'": "&apos;"})
        return (
            '<?xml version="1.0" encoding="UTF-8"?><Response><Gather input="speech" '
            'language="en-US" method="POST" speechTimeout="auto" timeout="8" '
            f'action="{callback}"><Say>{action}</Say></Gather><Say>Goodbye.</Say><Hangup/></Response>'
        )

    def _valid_signature(self, config, request_path, fields, signature):
        if not signature or not config["auth_token"] or not config["webhook_url"]:
            return False
        url = config["webhook_url"].rstrip("/") + request_path
        material = [url]
        for key in sorted(fields):
            for value in sorted(fields[key]):
                material.extend((key, value))
        expected = base64.b64encode(hmac.new(
            config["auth_token"].encode("utf-8"),
            "".join(material).encode("utf-8"), hashlib.sha1).digest()).decode("ascii")
        return hmac.compare_digest(expected, signature)

    def _api(self, endpoint, fields, notify_errors=True):
        with self._lock:
            config = dict(self._config)
        if not config["enabled"] or not config["account_sid"] or not config["auth_token"]:
            raise RuntimeError("Twilio is not enabled or configured.")
        url = f"https://api.twilio.com/2010-04-01/Accounts/{config['account_sid']}/{endpoint}"
        credentials = base64.b64encode(
            f"{config['account_sid']}:{config['auth_token']}".encode("utf-8")).decode("ascii")
        request = urllib.request.Request(
            url, data=urllib.parse.urlencode(fields).encode("utf-8"),
            headers={"Authorization": f"Basic {credentials}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            if notify_errors and exc.code >= 500:
                self._event(f"Twilio API connection failed: HTTP {exc.code}.")
            raise RuntimeError(f"Twilio request failed: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if notify_errors:
                self._event(f"Twilio API connection failed: {type(exc).__name__}.")
            raise RuntimeError(f"Twilio request failed: {type(exc).__name__}") from exc
        except ValueError as exc:
            raise RuntimeError("Twilio returned an invalid response.") from exc
        return result

    def send_connection_alert(self, message):
        with self._lock:
            config = dict(self._config)
        recipient = config["connection_alert_number"]
        if not config["enabled"] or not recipient:
            raise RuntimeError("Twilio connection alerts are not configured.")
        message = str(message or "").strip()
        if not message or len(message) > 1500:
            raise ValueError("Connection alert text must contain 1 to 1500 characters.")
        self._api("Messages.json", {
            "From": config["phone_number"], "To": recipient, "Body": message,
        }, notify_errors=False)

    def send_sms(self, recipient, message):
        with self._lock:
            config = dict(self._config)
        if not config["enabled"] or not config["outbound_sms"]:
            raise RuntimeError("Outbound SMS is disabled in Twilio settings.")
        return self._send_sms(recipient, message, permission="outbound_sms")

    def _send_sms(self, recipient, message, permission):
        with self._lock:
            config = dict(self._config)
        if not config["enabled"] or not config[permission]:
            raise RuntimeError(f"Twilio {permission.replace('_', ' ')} is disabled.")
        if recipient not in config["allowed_numbers"]:
            raise ValueError("That number is not on the Twilio allowlist.")
        message = str(message or "").strip()
        if not message or len(message) > 1500:
            raise ValueError("SMS text must contain 1 to 1500 characters.")
        self._api("Messages.json", {
            "From": config["phone_number"], "To": recipient, "Body": message,
        })
        self._event("Sent an outbound SMS to an allowlisted number.")
        return "Text message sent."

    def place_call(self, recipient, message):
        with self._lock:
            config = dict(self._config)
        if not config["enabled"] or not config["outbound_voice"]:
            raise RuntimeError("Outbound voice calls are disabled in Twilio settings.")
        if recipient not in config["allowed_numbers"]:
            raise ValueError("That number is not on the Twilio allowlist.")
        message = str(message or "").strip()
        if not message or len(message) > 1200:
            raise ValueError("Call message must contain 1 to 1200 characters.")
        self._api("Calls.json", {
            "From": config["phone_number"], "To": recipient,
            "Twiml": f"<Response><Say>{_xml_text(message)}</Say></Response>",
        })
        self._event("Started an outbound voice call to an allowlisted number.")
        return "Voice call started."


def _xml_text(value):
    value = str(value or "")
    valid = "".join(
        char for char in value
        if char in "\t\n\r"
        or "\u0020" <= char <= "\ud7ff"
        or "\ue000" <= char <= "\ufffd"
        or "\U00010000" <= char <= "\U0010ffff"
    )
    return escape(valid)
