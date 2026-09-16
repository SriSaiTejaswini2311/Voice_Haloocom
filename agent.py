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
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    ErrorEvent,
    JobContext,
    JobProcess,
    RunContext,
    UserStateChangedEvent,
    cli,
    function_tool,
    inference,
)
from livekit.agents.llm import ChatContext, ChatMessage, FallbackAdapter, StopResponse
from livekit.plugins import silero

from mock_data import lookup_verification_status
from prompts import get_system_prompt

load_dotenv()

# No logging.basicConfig() here on purpose: it installed a second root handler
# alongside the one LiveKit sets up in cli.run_app, so every LiveKit and app
# record was printed twice. LiveKit's logging system now owns root and our
# logger simply propagates to it - every logger.error/exception/info below still
# emits, at LiveKit's level and format.
logger = logging.getLogger("verification-agent")

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
        _emit(_tool_block(record))
        return record

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Error path 2: empty/whitespace STT transcript -> skip the LLM call."""
        text = (new_message.text_content or "").strip()
        if text:
            return
        _emit(f"{EMPTY_TRANSCRIPT_MARKER}  (empty STT transcript - LLM call skipped)")
        await self.session.say(EMPTY_TRANSCRIPT_PROMPT)
        raise StopResponse()


# ---------------------------------------------------------------------------
# Console presentation helpers
#
# Formatting only: these helpers never touch the pipeline. They replace the
# previous logger.info() status/turn/tool lines so each application event is
# printed exactly once and without the logging prefix, which keeps the terminal
# readable during a demo. LiveKit's own logging is left completely alone, and
# every error/exception still goes through the logger below.
# ---------------------------------------------------------------------------
LINE = "=" * 50
THIN = "-" * 50


def _encodable(text: str) -> bool:
    """True when the active console can render ``text``.

    Windows consoles are often cp1252/cp437, which cannot encode characters
    such as the tick and cross used by the status box.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _glyph(unicode_text: str, ascii_text: str) -> str:
    """Use the prettier glyph where the console supports it, else plain ASCII."""
    return unicode_text if _encodable(unicode_text) else ascii_text


DASH = _glyph("—", "-")
EMPTY_TRANSCRIPT_MARKER = f"[No speech detected {DASH} reprompting]"
INTERRUPTED_MARKER = f"[INTERRUPTED {DASH} user started speaking]"


def _emit(block: str) -> None:
    """Print one presentation block to stdout, flushed so it shows up live.

    Never raises: if the console cannot encode a character (spoken text can be
    in any script) the character is replaced instead of breaking the turn.
    """
    try:
        print(block, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        safe = block.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
        print(safe, flush=True)


def _banner(room: str) -> str:
    """Startup banner: assistant identity, models in use and the job's room."""
    return "\n".join(
        [
            LINE,
            "     CUSTOMER VERIFICATION ASSISTANT",
            LINE,
            "Agent:  Aria (customer-verification-agent)",
            f"Room:   {room}",
            f"STT:    LiveKit Inference ({STT_MODEL})",
            f"LLM:    LiveKit Inference ({LLM_MODEL})",
            f"TTS:    LiveKit Inference ({TTS_MODEL})",
            "Status: Ready",
            LINE,
        ]
    )


def _turn_block(role: str, text: str) -> str:
    """One conversation turn: USER/ARIA label followed by the spoken text."""
    speaker = {"user": "USER", "assistant": "ARIA"}.get(role, role.upper())
    return f"{speaker}:\n> {text}"


def _tool_block(record: dict[str, Any]) -> str:
    """Readable rendering of the record returned by the status tool."""
    lines = [
        THIN,
        "VERIFICATION CHECK",
        THIN,
        f"Customer: {record.get('customer_name') or '<unknown>'}",
    ]
    if record.get("found"):
        yes, no = _glyph("✓", "[OK]"), _glyph("✗", "[--]")
        pan = f"{yes} verified" if record.get("pan_verified") else f"{no} not verified"
        bank = (
            f"{yes} verified" if record.get("bank_verified") else f"{no} not verified"
        )
        selfie = (
            f"{yes} uploaded" if record.get("selfie_uploaded") else f"{no} not uploaded"
        )
        lines += [f"PAN:      {pan}", f"Bank:     {bank}", f"Selfie:   {selfie}"]
    else:
        lines.append("Status:   no record found")
    lines.append(THIN)
    return "\n".join(lines)


def _session_block(state: SessionState, reason: Any) -> str:
    """Session-end summary built only from state that was already collected."""
    interruptions = sum(1 for turn in state.turns if turn.get("interrupted"))
    tool_lookups = 1 if state.tool_lookup else 0
    return "\n".join(
        [
            LINE,
            "SESSION COMPLETE",
            LINE,
            f"Turns: {len(state.turns)}   Interruptions: {interruptions}   "
            f"Tool lookups: {tool_lookups}",
            f"Reason: {reason}",
            LINE,
        ]
    )


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

    _emit(
        f"[LATENCY]  e2e={as_ms('e2e_latency')}  "
        f"llm_ttft={as_ms('llm_node_ttft')}  tts_ttfb={as_ms('tts_node_ttfb')}"
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


# The LiveKit CLI (`lk agent console`, `lk agent dev`, `lk agent deploy`)
# discovers the agent through the module-level ``server`` variable, so the agent
# is registered on an AgentServer instead of being handed to run_app as
# WorkerOptions. Everything below the wiring is unchanged.
server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Load the VAD once per worker process so jobs start faster."""
    proc.userdata["vad"] = silero.VAD.load()


# Prewarm hook for the AgentServer (runs once per worker process).
server.setup_fnc = prewarm


@server.rtc_session(agent_name="customer-verification-agent")
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    _emit(_banner(ctx.room.name))

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
        if interrupted:
            _emit(INTERRUPTED_MARKER)
        _emit(_turn_block(role, text))

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
        _emit(_session_block(state, getattr(ev, "reason", ev)))
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
            _emit(
                f"[No speech detected {DASH} ending session]  "
                f"(no answer after {state.away_prompts} check-ins)"
            )
            session.shutdown()
            return

        state.away_prompts += 1
        _emit(
            f"{EMPTY_TRANSCRIPT_MARKER}  "
            f"(attempt {state.away_prompts}/{MAX_AWAY_PROMPTS})"
        )
        await session.generate_reply(instructions=AWAY_INSTRUCTIONS)
    except asyncio.CancelledError:
        # The customer spoke again - the check-in task is expected to cancel.
        raise
    except Exception:
        logger.exception("Inactivity re-prompt failed; session continues")


if __name__ == "__main__":
    cli.run_app(server)
