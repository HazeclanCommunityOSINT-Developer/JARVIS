"""Optional local face enrollment and recognition from the Motion MJPEG stream.

Requires opencv-contrib-python-headless. Face images and profiles stay in the user's
JARVIS config directory and are never added to the repository.
"""
import json
import os
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
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            return {}

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
            if os.path.isfile(self.model_path):
                self._recognizer = cv2.face.LBPHFaceRecognizer_create()
                self._recognizer.read(self.model_path)
            elif self._profiles:
                self._train()
            self._set_state(status="ready", message="Local face recognition is ready.")
        except Exception as exc:
            self._set_state(status="unavailable", message=str(exc))

    def _set_state(self, **changes):
        with self._lock:
            self._state.update(changes)
            self._state["profiles"] = self._profile_list()
            snapshot = json.loads(json.dumps(self._state))
        self.on_state(snapshot)
        for message in self._update_presence(snapshot):
            self.on_presence(message)

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
        self._thread = threading.Thread(target=self._run, name="jarvis-face-recognition", daemon=True)
        self._thread.start()
        self._set_state(status="connecting", message=f"Connecting to {self.stream_url}")
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
        self.start()
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
        with self._lock:
            profile = self._profiles.pop(str(profile_id), None)
            if not profile:
                raise ValueError("That face profile was not found.")
            if self._state.get("identity") == profile.get("name"):
                self._state.update(identity=None, profile_info="")
        import shutil
        shutil.rmtree(os.path.join(self.data_dir, str(profile_id)), ignore_errors=True)
        self._save_profiles()
        self._train()
        self._set_state(status="monitoring", message=f"Removed face profile for {profile.get('name')}.")

    def set_profile_role(self, profile_id, role):
        if role not in {"guest", "user", "admin"}:
            raise ValueError("Choose Guest, User, or Admin permissions.")
        with self._lock:
            profile = self._profiles.get(str(profile_id))
            if not profile:
                raise ValueError("That face profile was not found.")
            profile["role"] = role
        self._save_profiles()
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
        os.makedirs(self.data_dir, mode=0o700, exist_ok=True)
        tmp = self.profiles_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._profiles, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.profiles_path)

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
            capture = cv2.VideoCapture(self.stream_url)
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not capture.isOpened():
                    self._set_state(status="disconnected", message=f"Cannot open {self.stream_url}")
                    self._stop.wait(3)
                    continue
                self._set_state(status="monitoring", message="Watching the local Motion stream.")
                last_result = 0.0
                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    now = time.time()
                    with self._lock:
                        pending = self._pending
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
                    if crop is None or self._recognizer is None:
                        self._recognition_time = 0.0
                        self._set_state(status="monitoring", identity=None, profile_id=None,
                                        profile_role=None, profile_info="", last_seen=None,
                                        message="No enrolled face detected." if crop is None else "No face profiles enrolled.")
                        continue
                    label, distance = self._recognizer.predict(crop)
                    profile = next((p for p in self._profiles.values()
                                    if int(p.get("label", -1)) == int(label)), None)
                    if profile and float(distance) <= self.MATCH_DISTANCE:
                        self._recognition_time = time.monotonic()
                        profile_id = next((pid for pid, p in self._profiles.items() if p is profile), None)
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
                capture.release()
            self._stop.wait(2)

    def _finish_enrollment(self, pending):
        profile_dir = os.path.join(self.data_dir, pending["id"])
        os.makedirs(profile_dir, mode=0o700, exist_ok=True)
        for index, sample in enumerate(pending["samples"], start=1):
            self._cv2.imwrite(os.path.join(profile_dir, f"{index:03d}.png"), sample)
        with self._lock:
            self._profiles[pending["id"]] = {
                "name": pending["name"], "info": pending["info"], "role": pending["role"],
                "label": pending["label"],
            }
            self._pending = None
        self._save_profiles()
        self._train()
        self._set_state(status="monitoring", enrollment_count=0,
                        message=f"Enrollment complete for {pending['name']}.")

    def _train(self):
        if self._cv2 is None:
            return
        images, labels = [], []
        for profile_id, profile in self._profiles.items():
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
            self._recognizer = None
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            return
        recognizer = self._cv2.face.LBPHFaceRecognizer_create()
        import numpy as np
        recognizer.train(images, np.asarray(labels, dtype=np.int32))
        os.makedirs(self.data_dir, mode=0o700, exist_ok=True)
        recognizer.write(self.model_path)
        os.chmod(self.model_path, 0o600)
        self._recognizer = recognizer
