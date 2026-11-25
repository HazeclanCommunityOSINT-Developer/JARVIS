#!/bin/bash
# Motion -> JARVIS event log.
# Usage: jarvis-motion-event.sh <start|end|picture> <camera_number> [snapshot_path]
# Appends one JSON line per event. Override the log path with JARVIS_EVENTS if needed.
EVENTS="${JARVIS_EVENTS:-/var/lib/motion/jarvis-events.jsonl}"
ev="${1//[^a-z]/}"
cam="${2//[^0-9]/}"
file="${3//[\"\\]/}"
printf '{"t":%s,"event":"%s","cam":"%s","file":"%s"}\n' "$(date +%s)" "$ev" "$cam" "$file" >> "$EVENTS"
chmod 644 "$EVENTS" 2>/dev/null
# keep the log small: trim to the last 2000 lines once it passes ~2 MB
if [ "$(stat -c %s "$EVENTS" 2>/dev/null || echo 0)" -gt 2000000 ]; then
  tail -n 2000 "$EVENTS" > "$EVENTS.tmp" && mv "$EVENTS.tmp" "$EVENTS"
  chmod 644 "$EVENTS" 2>/dev/null
fi
exit 0
