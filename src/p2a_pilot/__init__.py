"""P2A → Gemini pilot toolkit.

See ``scripts/p2a_pilot/PILOT_PROTOCOL.md`` for the end-to-end workflow.

Modules:
  load_p2a       - parse P2A label JSONs, pick pilot videos, build anchor tables
  build_prompt   - generate the Gemini prompt text for one P2A video
  parse_response - parse + validate Gemini's JSON response against the schema
  accuracy       - compare Gemini output vs hand-truth (per-field accuracy report)
"""
