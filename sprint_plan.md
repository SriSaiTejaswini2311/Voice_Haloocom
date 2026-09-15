# Sprint Plan: Real-Time Voice Agent (Customer Verification Assistant)

> Companion to `implementation_plan.md`. The assignment must be completed in **3–4 hours total**,
> so this is a single-day, time-boxed micro-sprint plan (not a multi-week Agile sprint) with 5
> short sprints. Each sprint has a goal, tasks, time-box, and exit criteria (Definition of Done).
> An agent or developer can execute these sequentially and check items off as they go.

## Overview

| Sprint | Focus | Time-box | Cumulative |
|---|---|---|---|
| 0 | Environment & LiveKit Cloud setup | 20 min | 0:20 |
| 1 | Core STT→LLM→TTS pipeline (agent speaks) | 60 min | 1:20 |
| 2 | Conversation logic, prompt & error handling | 60 min | 2:20 |
| 3 | Bonus features (tool call, JSON, latency, config prompt) | 40 min | 3:00 |
| 4 | Local testing, deploy to LiveKit Cloud | 40 min | 3:40 |
| 5 | README, demo recording, production notes | 20 min | 4:00 |

---

## Sprint 0 — Environment & LiveKit Cloud Setup (20 min)

**Goal:** A runnable Python project skeleton connected to a LiveKit Cloud project.

- [ ] Create LiveKit Cloud account/project (free Build plan), no payment method added.
- [ ] Install LiveKit CLI (`lk`).
- [ ] Create project folder structure per `implementation_plan.md` §6.
- [ ] `pip install livekit-agents livekit-plugins-silero livekit-plugins-turn-detector python-dotenv`.
- [ ] Create `.env` (local only) with `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`;
      create `.env.example` with placeholders; add `.env` to `.gitignore`.
- [ ] `git init`, initial commit (skeleton + `.gitignore` + `.env.example`).

**Exit criteria:** `lk` CLI authenticated, dependencies installed, repo initialized, no secrets committed.

---

## Sprint 1 — Core Pipeline: Agent Speaks (60 min)

**Goal:** A minimal agent joins a room, greets the user, and completes one STT→LLM→TTS round trip.

- [ ] Write `agent.py` entrypoint using `AgentSession` with `inference.STT`, `inference.LLM`,
      `inference.TTS`, and `silero.VAD.load()` (see `implementation_plan.md` §7 step 4).
  Note: check the LATEST livekit inference model catalog (`docs.livekit.io/agents/models/inference`) 
  at build time — IDs shown in the plan are examples and may change.
- [ ] Register `agents.cli.run_app(...)` under `if __name__ == "__main__":`.
- [ ] Add a placeholder system prompt (hardcoded string) so the agent greets on session start.
- [ ] Run `python agent.py dev` and confirm the worker registers with LiveKit Cloud.
- [ ] Smoke test with `lk agent console` (terminal) — say "hello", hear a spoken reply.

**Exit criteria:** Worker registers ("registered worker" in logs); a full voice round trip
(mic → text → LLM reply → speech) works end-to-end in console mode.

---

## Sprint 2 — Conversation Logic, Prompt & Error Handling (60 min)

**Goal:** Full verification-assistant conversation flow + resilience to basic failure modes.

- [ ] Move system prompt into `prompts.py`; write the full instructions covering the 7-step flow
      and 3 FAQs (`implementation_plan.md` §3 and §7 step 2).
- [ ] Wire `prompts.SYSTEM_PROMPT` (with env-var override) into `agent.py`.
- [ ] Implement transcript logging: `conversation_item_added` handler prints/stores each turn.
- [ ] Implement error handling (§7 step 5):
  - [ ] No-speech-detected handling (timeout / gentle re-prompt).
  - [ ] Empty-transcription guard (skip LLM call, ask user to repeat).
  - [ ] Temporary model/API failure fallback (try/except or `FallbackAdapter` + graceful spoken message).
- [ ] Confirm interruption/barge-in works by default (`allow_interruptions=True`) — test by talking
      over the agent mid-response in console mode.
- [ ] Manually walk through the full script once: greeting → name → 3 verification Qs → 1 FAQ →
      summary → further-assistance check → closing.

**Exit criteria:** A full scripted conversation completes correctly; interrupting the agent mid-speech
works; at least one error scenario is demonstrably handled without crashing the process.

---

## Sprint 3 — Bonus Features (40 min, optional but time-permitting)

**Goal:** Add the "nice-to-have" items from the assignment if time allows. Skip any that threaten
the 4-hour budget — mandatory features always take priority.

- [ ] `mock_data.py` + `check_verification_status` `@function_tool` wired into the `Agent` (tool-call bonus).
- [ ] Latency logging: capture `ev.item.metrics` (`e2e_latency`, `llm_node_ttft`, `tts_node_ttfb`) and
      print/log per turn.
- [ ] Configurable system prompt via `AGENT_SYSTEM_PROMPT` env var (already scaffolded in Sprint 2).
- [ ] On session end, serialize transcript + verification summary to
      `transcripts/<room>-<timestamp>.json`.

**Exit criteria:** Whichever bonus items are attempted work correctly and don't destabilize the core flow.

---

## Sprint 4 — Local Testing & Cloud Deployment (40 min)

**Goal:** Agent is deployed to LiveKit Cloud and reachable from the browser-based Agent Console.

- [ ] Full regression pass in `lk agent console` (or dev mode) covering the Sprint 2 checklist again.
- [ ] Test via the browser **LiveKit Agent Console** (mic permissions, "Save and start session").
- [ ] Run `lk agent create` (or equivalent deploy command) to publish to LiveKit Cloud.
- [ ] Verify the deployed (cloud) worker — not just the local dev process — responds correctly from
      the Agent Console.
- [ ] Confirm no paid credits/payment method were required at any point.

**Exit criteria:** A cloud-deployed agent is demonstrably reachable and functional from the browser
Agent Console, matching local behavior.

---

## Sprint 5 — README, Demo Recording, Production Notes (20 min)

**Goal:** Package deliverables.

- [ ] Write `README.md` per `implementation_plan.md` §8 (setup, architecture, models used,
      interruption explanation, limitations, no-paid-services confirmation, tools used).
- [ ] Add "Production Improvements" section/note to README.
- [ ] Record 2–3 min screen capture: normal conversation, one interruption, transcript/logs visible.
- [ ] Final `git commit` + push; double-check `.env`/secrets are not tracked (`git status`, `.gitignore`).

**Exit criteria:** All items in `implementation_plan.md` §9 Deliverables Checklist are satisfied.

---

## Risk / Time-Management Notes

- If running over budget, cut in this order: Sprint 3 (bonus) → advanced error-handling variety in
  Sprint 2 (keep at least one) → polish in Sprint 5. Never cut Sprint 0, 1, 4 core deployment, or
  the mandatory README contents.
- Re-check current LiveKit Inference model IDs before Sprint 1 — provider model names/availability
  change over time; the ones listed in `implementation_plan.md` are illustrative, not guaranteed.
