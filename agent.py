"""Customer Verification Assistant - LiveKit Agents voice worker.

The entire STT -> LLM -> TTS pipeline runs on LiveKit Inference, so the only
credentials required are LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET.

Audio flow:
    browser mic
      -> LiveKit Cloud room (WebRTC ingress)
      -> Silero VAD + LiveKit turn detector   (end-of-utterance, barge-in)
      -> inference.STT                        (streaming speech -> text)
      -> inference.LLM                        (streaming chat + optional tool call)
      -> inference.TTS                        (streaming text -> speech)
      -> LiveKit Cloud room (WebRTC egress) -> browser speaker

The conversation script lives in prompts.py (the LLM follows an enumerated step
order using the chat history); this module only wires the pipeline, registers the
tool call, logs every turn + latency, and implements the error-handling paths.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    ErrorEvent,
    JobContext,
    JobProcess,
    RunContext,
    UserStateChangedEvent,
    function_tool,
    inference,
)
from livekit.agents.llm import ChatContext, ChatMessage, FallbackAdapter, StopResponse
from livekit.plugins import silero

from mock_data import lookup_verification_status
from prompts import get_system_prompt

load_dotenv()

logger = logging.getLogger("verification-agent")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Models - every one is served by LiveKit Inference, so no provider API keys
# and no provider plugins are needed.
#
# IDs verified against the model catalog shipped with livekit-agents 1.8.1
# (livekit.agents.inference STTModels / LLMModels / TTSModels) and against
# https://docs.livekit.io/agents/models/inference on 2026-09-12.
# Each value can be overridden with an environment variable.
# ---------------------------------------------------------------------------
STT_MODEL = os.getenv("STT_MODEL", "deepgram/nova-3")
STT_FALLBACK_MODEL = os.getenv("STT_FALLBACK_MODEL", "deepgram/flux-general")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "google/gemini-2.5-flash-lite")
TTS_MODEL = os.getenv("TTS_MODEL", "cartesia/sonic-3")
# Cartesia "Jacqueline" (en-US). Omit/blank TTS_VOICE to use the provider default.
TTS_VOICE = os.getenv("TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
TTS_FALLBACK_MODEL = os.getenv("TTS_FALLBACK_MODEL", "deepgram/aura-2")

# Error-handling tuning.
USER_AWAY_TIMEOUT = float(os.getenv("USER_AWAY_TIMEOUT", "12"))
MAX_AWAY_PROMPTS = int(os.getenv("MAX_AWAY_PROMPTS", "3"))

TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "transcripts"

# Spoken lines used by the error-handling paths.
GRACEFUL_FALLBACK = (
    "Sorry, I'm having a little trouble with that right now. "
    "Could you please say that again?"
)
EMPTY_TRANSCRIPT_PROMPT = "I didn't quite catch that. Could you please repeat what you said?"
AWAY_CHECKIN = "Are you still there? I'm here whenever you're ready to continue."

GREETING_INSTRUCTIONS = (
    "Greet the customer warmly, introduce yourself as the Customer Verification "
    "Assistant, and ask for their full name. Do not say anything else."
)
AWAY_INSTRUCTIONS = (
    "The customer has been silent for a while. Politely say something like: "
    f"'{AWAY_CHECKIN}' Then, in the same reply, continue from the last unfinished "
    "verification step, using the conversation history."
)


@dataclass
class SessionState:
    """Per-job state: transcript, latency samples and error-handling bookkeeping."""

    room_name: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    turns: list[dict[str, Any]] = field(default_factory=list)
    latency: list[dict[str, Any]] = field(default_factory=list)
    tool_lookup: dict[str, Any] | None = None
    away_task: asyncio.Task[None] | None = None
    away_prompts: int = 0


class VerificationAssistant(Agent):
    """Verification assistant: script from prompts.py, plus one tool and one guard."""

    def __init__(self, state: SessionState) -> None:
        super().__init__(instructions=get_system_prompt())
        self._state = state

    @function_tool()
    async def check_verification_status(
        self,
        context: RunContext,
        customer_name: str,
    ) -> dict[str, Any]:
        """Look up the customer's current verification status in the KYC system.

        Call this whenever the customer asks about their verification status, or
        asks whether a particular step (PAN, bank account, selfie) is already done.

        Args:
            customer_name: The customer's full name, exactly as they stated it.
        """
        record = lookup_verification_status(customer_name)
        self._state.tool_lookup = record
        logger.info("tool check_verification_status(%r) -> %s", customer_name, record)
        return record

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Error path 2: empty/whitespace STT transcript -> skip the LLM call."""
        text = (new_message.text_content or "").strip()
        if text:
            return
        logger.warning("Empty or whitespace STT transcript; skipping the LLM call")
        await self.session.say(EMPTY_TRANSCRIPT_PROMPT)
        raise StopResponse()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _metrics_dict(item: ChatMessage) -> dict[str, Any]:
    """Per-turn metrics attached to a ChatMessage (MetricsReport is a plain dict)."""
    metrics = getattr(item, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _log_latency(metrics: dict[str, Any]) -> None:
    """Bonus: log e2e_latency / llm_node_ttft / tts_node_ttfb for every reply."""

    def as_ms(key: str) -> str:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return f"{value * 1000:.0f}ms"
        return "n/a"

    logger.info(
        "LATENCY e2e_latency=%s llm_node_ttft=%s tts_node_ttfb=%s",
        as_ms("e2e_latency"),
        as_ms("llm_node_ttft"),
        as_ms("tts_node_ttfb"),
    )


# ---------------------------------------------------------------------------
# Verification summary extraction (customer name + the three answers)
# ---------------------------------------------------------------------------
_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|correct|already|done|completed|uploaded|verified|"
    r"i have|i did|i've)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"\b(no|nope|not yet|not done|haven't|have not|didn't|did not|pending|"
    r"still waiting|none)\b",
    re.IGNORECASE,
)
_TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "name": re.compile(r"\bname\b", re.IGNORECASE),
    "pan": re.compile(r"\bpan\b", re.IGNORECASE),
    "bank": re.compile(r"\bbank\b|\baccount\b", re.IGNORECASE),
    "selfie": re.compile(r"\bselfie\b|\bphoto\b", re.IGNORECASE),
}


def _normalize_yes_no(text: str | None) -> str:
    """Map free-form spoken text to yes / no / unclear / unknown."""
    if not text or not text.strip():
        return "unknown"
    if _NO_RE.search(text):
        return "no"
    if _YES_RE.search(text):
        return "yes"
    return "unclear"


def _bool_to_yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _normalize_yes_no(None if value is None else str(value))


_NAME_FILLERS = re.compile(
    r"^\s*(?:hi|hello|hey|yeah|yes|yep|sure|ok|okay|so|well|um|uh)?[\s,]*"
    r"(?:my\s+(?:full\s+)?name\s+is|my\s+name'?s|the\s+name\s+is|name\s+is|"
    r"i\s+am|i'm|im|it'?s|its|this\s+is|you\s+can\s+call\s+me|call\s+me)\s+",
    re.IGNORECASE,
)


def _clean_name(text: str | None) -> str | None:
    """Best-effort name cleanup: "My name is Priya Sharma." -> "Priya Sharma"."""
    if not text or not text.strip():
        return None
    stripped = text.strip()
    cleaned = _NAME_FILLERS.sub("", stripped).strip().strip(".,!?;:")
    return cleaned or stripped


def _pair_answers(turns: list[dict[str, Any]]) -> dict[str, str]:
    """Pair each assistant question topic with the customer's next spoken reply."""
    answers: dict[str, str] = {}
    for index, turn in enumerate(turns):
        if turn.get("role") != "assistant":
            continue
        text = turn.get("text") or ""
        for topic, pattern in _TOPIC_PATTERNS.items():
            if topic in answers or not pattern.search(text):
                continue
            for later in turns[index + 1 :]:
                reply = (later.get("text") or "").strip()
                if later.get("role") == "user" and reply:
                    answers[topic] = reply
                    break
    return answers


def _build_summary(state: SessionState) -> dict[str, Any]:
    """Customer name + the three verification answers.

    Prefers the authoritative record returned by ``check_verification_status``
    and otherwise pairs questions with answers in the transcript.
    """
    if state.tool_lookup and state.tool_lookup.get("found"):
        record = state.tool_lookup
        return {
            "source": "check_verification_status tool",
            "customer_name": record.get("customer_name"),
            "pan_verified": _bool_to_yes_no(record.get("pan_verified")),
            "bank_verified": _bool_to_yes_no(record.get("bank_verified")),
            "selfie_uploaded": _bool_to_yes_no(record.get("selfie_uploaded")),
        }

    answers = _pair_answers(state.turns)
    return {
        "source": "conversation transcript",
        "customer_name": _clean_name(answers.get("name")),
        "pan_verified": _normalize_yes_no(answers.get("pan")),
        "bank_verified": _normalize_yes_no(answers.get("bank")),
        "selfie_uploaded": _normalize_yes_no(answers.get("selfie")),
    }


def _write_session_json(state: SessionState) -> Path | None:
    """Bonus: write the transcript + summary to transcripts/<room>-<timestamp>.json."""
    if not state.turns:
        logger.info("No conversation turns recorded; skipping the transcript file")
        return None

    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_room = re.sub(r"[^A-Za-z0-9._-]+", "_", state.room_name) or "room"
    path = TRANSCRIPTS_DIR / f"{safe_room}-{stamp}.json"

    payload = {
        "room_name": state.room_name,
        "started_at": state.started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "stt": STT_MODEL,
            "stt_fallback": STT_FALLBACK_MODEL,
            "llm": LLM_MODEL,
            "llm_fallback": LLM_FALLBACK_MODEL,
            "tts": TTS_MODEL,
            "tts_voice": TTS_VOICE,
            "tts_fallback": TTS_FALLBACK_MODEL,
        },
        "summary": _build_summary(state),
        "transcript": state.turns,
        "latency": state.latency,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote session transcript and summary to %s", path)
    return path


def _build_session(vad: silero.VAD) -> AgentSession:
    """Wire streaming STT -> LLM -> TTS, all through LiveKit Inference."""
    return AgentSession(
        vad=vad,
        # Streaming STT with a server-side fallback model.
        stt=inference.STT(
            model=STT_MODEL,
            language="en",
            fallback=[STT_FALLBACK_MODEL],
        ),
        # inference.LLM has no built-in fallback, so use the framework adapter.
        llm=FallbackAdapter(
            llm=[
                inference.LLM(model=LLM_MODEL),
                inference.LLM(model=LLM_FALLBACK_MODEL),
            ],
            attempt_timeout=6.0,
            max_retry_per_llm=1,
        ),
        # Streaming TTS. A stale voice ID cannot break the demo: the server-side
        # fallback model speaks with its own provider default voice.
        tts=inference.TTS(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            language="en",
            fallback=[TTS_FALLBACK_MODEL],
        ),
        # Error path 1: quiet customer -> user state changes to "away".
        user_away_timeout=USER_AWAY_TIMEOUT,
        # Interruptions / barge-in: the customer may talk over the agent.
        allow_interruptions=True,
        turn_handling={
            # ML end-of-turn detection (uses LiveKit Inference, local fallback).
            "turn_detection": inference.TurnDetector(),
            "interruption": {"enabled": True},
        },
    )


def prewarm(proc: JobProcess) -> None:
    """Load the VAD once per worker process so jobs start faster."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("Job assigned for room %s", ctx.room.name)

    state = SessionState(room_name=ctx.room.name)
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    session = _build_session(vad)

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        """Log every turn (stdout) and collect per-turn latency metrics."""
        item = ev.item
        if not isinstance(item, ChatMessage):
            return

        role = item.role
        text = item.text_content or ""
        interrupted = bool(getattr(item, "interrupted", False))
        logger.info(
            "TURN role=%s interrupted=%s text=%r", role, interrupted, text
        )

        state.turns.append(
            {
                "role": role,
                "text": text,
                "interrupted": interrupted,
                "created_at": getattr(item, "created_at", None),
            }
        )

        if role == "assistant":
            metrics = _metrics_dict(item)
            state.latency.append(
                {
                    "e2e_latency": metrics.get("e2e_latency"),
                    "llm_node_ttft": metrics.get("llm_node_ttft"),
                    "tts_node_ttfb": metrics.get("tts_node_ttfb"),
                }
            )
            _log_latency(metrics)

    @session.on("user_state_changed")
    def on_user_state_changed(ev: UserStateChangedEvent) -> None:
        """Error path 1: the customer went quiet -> start the 'still there?' timer."""
        if ev.new_state == "away":
            if state.away_task is None or state.away_task.done():
                state.away_task = asyncio.create_task(
                    _prompt_if_away(session, state)
                )
            return

        # Any other state means the customer is active again: stop the check-in.
        if state.away_task is not None:
            state.away_task.cancel()
            state.away_task = None
        if ev.new_state == "speaking":
            # Reset the strike counter only when the customer actually speaks.
            state.away_prompts = 0

    @session.on("error")
    def on_error(ev: ErrorEvent) -> None:
        """Error path 3: log the failure and speak a graceful fallback line."""
        error = ev.error
        logger.error(
            "Pipeline error from %s (recoverable=%s): %s",
            type(ev.source).__name__,
            getattr(error, "recoverable", None),
            error,
        )
        try:
            session.say(GRACEFUL_FALLBACK)
        except Exception:
            logger.exception("Failed to speak the graceful fallback line")
        # Mark it handled so the session does not tear the job down.
        if hasattr(error, "recoverable"):
            error.recoverable = True

    @session.on("close")
    def on_close(ev: Any) -> None:
        logger.info("Session closed (reason=%s)", getattr(ev, "reason", ev))
        if state.away_task is not None:
            state.away_task.cancel()
        try:
            _write_session_json(state)
        except Exception:
            logger.exception("Failed to write the session transcript JSON")

    # session.start connects the room itself when room I/O is used.
    await session.start(agent=VerificationAssistant(state), room=ctx.room)
    # Kick off the script: greet the customer and ask for their name.
    await session.generate_reply(instructions=GREETING_INSTRUCTIONS)


async def _prompt_if_away(session: AgentSession, state: SessionState) -> None:
    """Error path 1: re-prompt a silent customer instead of hanging.

    Runs at most ``MAX_AWAY_PROMPTS`` times, then shuts the session down
    gracefully (closing the room) instead of waiting forever.
    """
    try:
        if state.away_prompts >= MAX_AWAY_PROMPTS:
            logger.info(
                "Customer still silent after %s check-ins; ending the session",
                state.away_prompts,
            )
            session.shutdown()
            return

        state.away_prompts += 1
        logger.info(
            "No speech detected; re-prompting (attempt %s/%s)",
            state.away_prompts,
            MAX_AWAY_PROMPTS,
        )
        await session.generate_reply(instructions=AWAY_INSTRUCTIONS)
    except asyncio.CancelledError:
        # The customer spoke again - the check-in task is expected to cancel.
        raise
    except Exception:
        logger.exception("Inactivity re-prompt failed; session continues")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="customer-verification-agent",
        )
    )
