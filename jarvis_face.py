"""Optional local face enrollment and recognition from the Motion MJPEG stream.

Requires opencv-contrib-python-headless. Face images and profiles stay in the user's
JARVIS config directory and are never added to the repository.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid


class FaceRecognitionService:
    SAMPLE_COUNT = 20
    SAMPLE_INTERVAL = 0.35
    MATCH_DISTANCE = 75.0

    def __init__(self, stream_url, data_dir, on_state=None, on_presence=None,
                 departure_grace=90, arrival_stable=2.0):
        self.stream_url = stream_url
        self.data_dir = os.path.join(data_dir, "faces")
        self.profiles_path = os.path.join(self.data_dir, "profiles.json")
        self.model_path = os.path.join(self.data_dir, "recognizer.yml")
        self.on_state = on_state or (lambda _state: None)
        self.on_presence = on_presence or (lambda _message: None)
        self.departure_grace = departure_grace
        self.arrival_stable = arrival_stable
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._cv2 = None
        self._cascade = None
        self._recognizer = None
        self._pending = None
        self._recognition_time = 0.0
        self._present = {}
        self._candidate_id = None
        self._candidate_since = 0.0
        self._profile_store_blocked = False
        self._profiles = self._load_profiles()
        self._state = {
            "status": "disabled", "identity": None, "profile_info": "",
            "profiles": self._profile_list(), "enrollment_count": 0,
            "enrollment_target": self.SAMPLE_COUNT, "message": "Recognition is not started.",
            "last_seen": None,
        }
        self._load_cv()

    def _load_profiles(self):
        try:
            with open(self.profiles_path, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return {}
        except OSError as exc:
            self._profile_store_blocked = True
            print(f"[face] Could not load profiles.json: {type(exc).__name__}: {exc}")
            return {}
        except (UnicodeError, json.JSONDecodeError) as exc:
            self._preserve_invalid_profiles(f"invalid JSON ({type(exc).__name__})")
            return {}
        if not isinstance(raw, dict):
            self._preserve_invalid_profiles("unexpected JSON structure")
            print("[face] profiles.json must contain an object; ignoring invalid profile data")
            return {}
        profiles = {}
        for profile_id, profile in raw.items():
            valid_id = (isinstance(profile_id, str)
                        and re.fullmatch(r"[0-9a-f]{32}", profile_id))
            valid_profile = (
                isinstance(profile, dict)
                and isinstance(profile.get("name"), str) and bool(profile["name"].strip())
                and isinstance(profile.get("info", ""), str)
                and isinstance(profile.get("role", "user"), str)
                and profile.get("role", "user") in {"guest", "user", "admin"}
                and isinstance(profile.get("label"), int)
                and not isinstance(profile.get("label"), bool)
                and profile["label"] > 0
            )
            if valid_id and valid_profile:
                profile.setdefault("role", "user")
                profiles[profile_id] = profile
            else:
                print(f"[face] Ignoring malformed profile entry {profile_id!r}")
        if len(profiles) != len(raw):
            self._preserve_invalid_profiles("malformed profile entries")
        return profiles

    def _preserve_invalid_profiles(self, reason):
        backup = f"{self.profiles_path}.corrupt-{uuid.uuid4().hex}"
        try:
            shutil.copy2(self.profiles_path, backup)
        except OSError as exc:
            self._profile_store_blocked = True
            print(f"[face] Could not preserve profiles.json after {reason}: {exc}")
        else:
            self._profile_store_blocked = False
            print(f"[face] Preserved invalid profile data at {backup} ({reason}).")

    def _profile_list(self):
        return [{"id": key, "name": item.get("name", ""), "info": item.get("info", ""),
                 "role": item.get("role", "user")}
                for key, item in self._profiles.items() if isinstance(item, dict)]

    def _load_cv(self):
        try:
            import cv2
            if not hasattr(cv2, "face"):
                raise RuntimeError("OpenCV face module is missing; install opencv-contrib-python-headless")
            self._cv2 = cv2
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self._cascade = cv2.CascadeClassifier(cascade_path)
            if self._cascade.empty():
                raise RuntimeError("OpenCV face detection data could not be loaded")
            if self._profile_store_blocked:
                raise RuntimeError("Profile data could not be read; fix profiles.json before enabling recognition.")
            if os.path.isfile(self.model_path):
                self._recognizer = cv2.face.LBPHFaceRecognizer_create()
                try:
                    self._recognizer.read(self.model_path)
                except Exception as exc:
                    print(f"[face] Could not load recognizer model: {type(exc).__name__}: {exc}; rebuilding")
                    self._recognizer = None
                    self._train()
            elif self._profiles:
                self._train()
            self._set_state(status="ready", message="Local face recognition is ready.")
        except Exception as exc:
            self._cv2 = None
            self._cascade = None
            self._recognizer = None
            self._set_state(status="unavailable", message=str(exc))

    def _set_state(self, **changes):
        with self._lock:
            self._state.update(changes)
            self._state["profiles"] = self._profile_list()
            snapshot = json.loads(json.dumps(self._state))
        try:
            self.on_state(snapshot)
        except Exception as exc:
            print(f"[face] State callback failed: {type(exc).__name__}: {exc}")
        try:
            messages = self._update_presence(snapshot)
        except Exception as exc:
            print(f"[face] Presence tracking failed: {type(exc).__name__}: {exc}")
            return
        for message in messages:
            try:
                self.on_presence(message)
            except Exception as exc:
                print(f"[face] Presence callback failed: {type(exc).__name__}: {exc}")

    def set_departure_grace(self, seconds):
        with self._lock:
            self.departure_grace = max(30, min(600, int(seconds)))

    def _update_presence(self, state):
        now = time.monotonic()
        events = []
        with self._lock:
            status = state.get("status")
            if status not in {"recognized", "monitoring", "unknown"}:
                # Pause absence timers during camera outages, enrollment, or a manual stop.
                for present in self._present.values():
                    present["last_seen"] = now
                self._candidate_id = None
                self._candidate_since = 0.0
                return events

            profile_id = state.get("profile_id") if status == "recognized" else None
            if profile_id:
                if profile_id != self._candidate_id:
                    self._candidate_id = profile_id
                    self._candidate_since = now
                present = self._present.get(profile_id)
                if present:
                    present["name"] = state.get("identity") or present["name"]
                    present["last_seen"] = now
                elif now - self._candidate_since >= self.arrival_stable:
                    name = state.get("identity") or "there"
                    self._present[profile_id] = {"name": name, "last_seen": now}
                    events.append(f"Welcome back, {name}.")
            else:
                self._candidate_id = None
                self._candidate_since = 0.0

            for old_id, present in list(self._present.items()):
                if now - present["last_seen"] >= self.departure_grace:
                    events.append(f"Goodbye, {present['name']}.")
                    del self._present[old_id]
        return events

    def snapshot(self):
        with self._lock:
            return json.loads(json.dumps(self._state))

    def start(self):
        if self._cv2 is None:
            return False
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._set_state(status="connecting", message=f"Connecting to {self.stream_url}")
        self._thread = threading.Thread(target=self._run, name="jarvis-face-recognition", daemon=True)
        try:
            self._thread.start()
        except Exception as exc:
            self._thread = None
            self._set_state(status="error", message=f"Could not start recognition: {type(exc).__name__}: {exc}")
            return False
        return True

    def stop(self):
        self._stop.set()
        with self._lock:
            self._pending = None
        self._set_state(status="stopped", identity=None, profile_info="", message="Recognition stopped.")

    def start_enrollment(self, name, info="", role="user"):
        if not isinstance(name, str) or not isinstance(info, str):
            raise ValueError("Profile name and user information must be text.")
        if role not in {"guest", "user", "admin"}:
            raise ValueError("Choose Guest, User, or Admin permissions.")
        name = " ".join(str(name).strip().split())[:80]
        info = " ".join(str(info).strip().split())[:500]
        if not name:
            raise ValueError("Enter a name for this face profile.")
        if self._cv2 is None:
            raise RuntimeError("Face recognition is unavailable; install opencv-contrib-python-headless.")
        if not self.start():
            raise RuntimeError("Face recognition could not start; check the dashboard status.")
        with self._lock:
            if self._pending:
                raise RuntimeError("An enrollment is already in progress.")
            used_labels = [int(p.get("label", 0)) for p in self._profiles.values() if isinstance(p, dict)]
            profile_id = uuid.uuid4().hex
            self._pending = {"id": profile_id, "label": max(used_labels, default=0) + 1,
                             "name": name, "info": info, "role": role,
                             "samples": [], "last_sample": 0.0}
        self._set_state(status="enrolling", identity=None, profile_info="", enrollment_count=0,
                        message=f"Enrolling {name}: look at the camera and move your head slightly.")

    def cancel_enrollment(self):
        with self._lock:
            self._pending = None
        self._set_state(status="monitoring", enrollment_count=0, message="Enrollment cancelled.")

    def delete_profile(self, profile_id):
        profile_id = str(profile_id)
        profile_dir = os.path.join(self.data_dir, profile_id)
        quarantine = None
        with self._lock:
            profile = self._profiles.get(profile_id)
            if not profile:
                raise ValueError("That face profile was not found.")
            if os.path.lexists(profile_dir):
                quarantine = os.path.join(self.data_dir,
                                          f".{profile_id}.deleting-{uuid.uuid4().hex}")
                os.replace(profile_dir, quarantine)
            self._profiles.pop(profile_id)
            try:
                self._save_profiles()
            except Exception:
                self._profiles[profile_id] = profile
                if quarantine:
                    try:
                        os.replace(quarantine, profile_dir)
                    except OSError as restore_exc:
                        print(f"[face] Could not restore profile samples after save failure: {restore_exc}")
                raise
            if self._state.get("identity") == profile.get("name"):
                self._state.update(identity=None, profile_info="")
        warning = ""
        if quarantine:
            try:
                if os.path.islink(quarantine):
                    os.unlink(quarantine)
                else:
                    shutil.rmtree(quarantine)
            except OSError as exc:
                warning = " Profile metadata was removed, but sample cleanup failed."
                print(f"[face] Could not remove samples for {profile.get('name')}: {exc}")
        try:
            self._train()
        except Exception as exc:
            warning += " Recognition retraining failed; restart JARVIS to retry."
            print(f"[face] Retraining after profile deletion failed: {type(exc).__name__}: {exc}")
        status = "error" if warning else "monitoring"
        message = f"Removed face profile for {profile.get('name')}.{warning}"
        self._set_state(status=status, message=message)
        return message

    def set_profile_role(self, profile_id, role):
        if role not in {"guest", "user", "admin"}:
            raise ValueError("Choose Guest, User, or Admin permissions.")
        with self._lock:
            profile = self._profiles.get(str(profile_id))
            if not profile:
                raise ValueError("That face profile was not found.")
            previous_role = profile.get("role", "user")
            profile["role"] = role
            try:
                self._save_profiles()
            except Exception:
                profile["role"] = previous_role
                raise
        self._set_state(message=f"Updated {profile.get('name')} permissions to {role}.")

    def current_profile(self, max_age=3.0):
        with self._lock:
            if self._cv2 is None:
                return None
            status = self._state.get("status")
            if status == "recognized" and time.monotonic() - self._recognition_time <= max_age:
                profile_id = self._state.get("profile_id")
                profile = self._profiles.get(profile_id)
                if profile:
                    return {"id": profile_id, "name": profile.get("name", ""),
                            "role": profile.get("role", "user")}
            # A working recognizer that has no recent match assigns the restricted guest role.
            if status in {"connecting", "monitoring", "unknown", "enrolling", "disconnected", "stopped", "error"}:
                return {"id": None, "name": "", "role": "guest"}
            return None

    def _save_profiles(self):
        if self._profile_store_blocked:
            raise RuntimeError("Profile data could not be read; fix profiles.json before changing profiles.")
        os.makedirs(self.data_dir, mode=0o700, exist_ok=True)
        fd = None
        tmp = None
        try:
            with self._lock:
                profiles = {profile_id: dict(profile)
                            for profile_id, profile in self._profiles.items()}
            fd, tmp = tempfile.mkstemp(prefix=".profiles.", suffix=".tmp", dir=self.data_dir)
            os.fchmod(fd, 0o600)
            output = os.fdopen(fd, "w", encoding="utf-8")
            fd = None
            with output as f:
                json.dump(profiles, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.profiles_path)
            tmp = None
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError as exc:
                    print(f"[face] Temporary profile file cleanup failed: {exc}")

    def _face_crop(self, frame):
        cv2 = self._cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5,
                                               minSize=(70, 70))
        if len(faces) != 1:
            return None
        x, y, w, h = faces[0]
        crop = gray[y:y + h, x:x + w]
        return cv2.resize(crop, (200, 200), interpolation=cv2.INTER_AREA)

    def _run(self):
        cv2 = self._cv2
        while not self._stop.is_set():
            capture = None
            try:
                capture = cv2.VideoCapture(self.stream_url)
                if capture is None:
                    raise RuntimeError("OpenCV could not create the video capture.")
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    self._set_state(status="disconnected", message=f"Cannot open {self.stream_url}")
                    self._stop.wait(3)
                    continue
                if self._profiles and self._recognizer is None:
                    self._train()
                self._set_state(status="monitoring", message="Watching the local Motion stream.")
                last_result = 0.0
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        self._recognition_time = 0.0
                        self._set_state(status="disconnected", identity=None, profile_id=None,
                                        profile_role=None, profile_info="",
                                        message="Motion stream stopped; reconnecting.")
                        break
                    now = time.time()
                    with self._lock:
                        pending = self._pending
                        recognizer = self._recognizer
                        profiles = list(self._profiles.items())
                    if pending:
                        crop = self._face_crop(frame)
                        if crop is not None and now - pending["last_sample"] >= self.SAMPLE_INTERVAL:
                            pending["last_sample"] = now
                            pending["samples"].append(crop)
                            count = len(pending["samples"])
                            self._set_state(status="enrolling", enrollment_count=count,
                                            message=f"Enrolling {pending['name']}: {count}/{self.SAMPLE_COUNT} samples.")
                            if count >= self.SAMPLE_COUNT:
                                self._finish_enrollment(pending)
                        continue
                    if now - last_result < 0.5:
                        continue
                    last_result = now
                    crop = self._face_crop(frame)
                    if crop is None or recognizer is None:
                        self._recognition_time = 0.0
                        self._set_state(status="monitoring", identity=None, profile_id=None,
                                        profile_role=None, profile_info="", last_seen=None,
                                        message=("No enrolled face detected." if crop is None else
                                                 "No face profiles enrolled." if not profiles else
                                                 "Face profiles are loaded, but the recognizer is unavailable."))
                        continue
                    label, distance = recognizer.predict(crop)
                    profile = next((p for _, p in profiles
                                    if int(p.get("label", -1)) == int(label)), None)
                    if profile and float(distance) <= self.MATCH_DISTANCE:
                        self._recognition_time = time.monotonic()
                        profile_id = next((pid for pid, p in profiles if p is profile), None)
                        self._set_state(status="recognized", identity=profile.get("name"),
                                        profile_id=profile_id, profile_role=profile.get("role", "user"),
                                        profile_info=profile.get("info", ""), last_seen=time.strftime("%H:%M:%S"),
                                        message=f"Local match distance {float(distance):.1f} (lower is closer).")
                    else:
                        self._recognition_time = 0.0
                        self._set_state(status="unknown", identity=None, profile_id=None,
                                        profile_role=None, profile_info="", last_seen=time.strftime("%H:%M:%S"),
                                        message="Face detected; no enrolled profile matched.")
            except Exception as exc:
                self._set_state(status="error", message=f"Face recognition error: {type(exc).__name__}: {exc}")
            finally:
                if capture is not None:
                    try:
                        capture.release()
                    except Exception as exc:
                        print(f"[face] Video capture cleanup failed: {type(exc).__name__}: {exc}")
            self._stop.wait(2)
        self._set_state(status="stopped", identity=None, profile_id=None,
                        profile_role=None, profile_info="", message="Recognition stopped.")

    def _finish_enrollment(self, pending):
        profile_dir = os.path.join(self.data_dir, pending["id"])
        try:
            os.makedirs(profile_dir, mode=0o700, exist_ok=False)
            for index, sample in enumerate(pending["samples"], start=1):
                output = os.path.join(profile_dir, f"{index:03d}.png")
                if not self._cv2.imwrite(output, sample):
                    raise OSError(f"OpenCV could not write enrollment sample {index}.")
        except Exception:
            with self._lock:
                if self._pending is pending:
                    self._pending = None
            try:
                shutil.rmtree(profile_dir)
            except FileNotFoundError:
                pass
            except OSError as cleanup_exc:
                print(f"[face] Could not remove incomplete enrollment samples: {cleanup_exc}")
            raise

        with self._lock:
            if self._pending is not pending:
                shutil.rmtree(profile_dir)
                return
            self._profiles[pending["id"]] = {
                "name": pending["name"], "info": pending["info"], "role": pending["role"],
                "label": pending["label"],
            }
            self._pending = None
            try:
                self._save_profiles()
            except Exception:
                self._profiles.pop(pending["id"], None)
                try:
                    shutil.rmtree(profile_dir)
                except OSError as cleanup_exc:
                    print(f"[face] Could not remove unsaved enrollment samples: {cleanup_exc}")
                raise
        try:
            self._train()
        except Exception as exc:
            message = (f"Enrollment saved for {pending['name']}, but recognition training failed: "
                       f"{type(exc).__name__}: {exc}. Restart JARVIS to retry.")
            print(f"[face] {message}")
            self._set_state(status="error", enrollment_count=0, message=message)
            return
        self._set_state(status="monitoring", enrollment_count=0,
                        message=f"Enrollment complete for {pending['name']}.")

    def _train(self):
        if self._cv2 is None:
            return
        with self._lock:
            profiles = {profile_id: dict(profile)
                        for profile_id, profile in self._profiles.items()}
            self._recognizer = None
        try:
            os.unlink(self.model_path)
        except FileNotFoundError:
            pass
        images, labels = [], []
        for profile_id, profile in profiles.items():
            directory = os.path.join(self.data_dir, profile_id)
            if not os.path.isdir(directory):
                continue
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                image = self._cv2.imread(path, self._cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    images.append(self._cv2.resize(image, (200, 200)))
                    labels.append(int(profile["label"]))
        if not images:
            return
        recognizer = self._cv2.face.LBPHFaceRecognizer_create()
        import numpy as np
        recognizer.train(images, np.asarray(labels, dtype=np.int32))
        os.makedirs(self.data_dir, mode=0o700, exist_ok=True)
        fd = None
        temporary_model = None
        try:
            fd, temporary_model = tempfile.mkstemp(prefix=".recognizer.", suffix=".yml",
                                                   dir=self.data_dir)
            os.close(fd)
            fd = None
            recognizer.write(temporary_model)
            os.chmod(temporary_model, 0o600)
            os.replace(temporary_model, self.model_path)
            temporary_model = None
            with self._lock:
                self._recognizer = recognizer
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if temporary_model:
                try:
                    os.unlink(temporary_model)
                except OSError as exc:
                    print(f"[face] Temporary recognizer cleanup failed: {exc}")
