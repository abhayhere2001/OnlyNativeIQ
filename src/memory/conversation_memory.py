from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable
from uuid import uuid4

class ConversationMemoryError(RuntimeError): pass
class InvalidConversationRoleError(ConversationMemoryError): pass

class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

@dataclass(frozen=True, slots=True)
class ConversationTurn:
    turn_id: str
    role: ConversationRole
    content: str
    created_at: datetime

    @classmethod
    def create(cls, role: ConversationRole | str, content: str) -> "ConversationTurn":
        if isinstance(role, str):
            try:
                role = ConversationRole(role.strip().lower())
            except ValueError as exc:
                raise InvalidConversationRoleError(f"Unsupported conversation role '{role}'.") from exc
        content = (content or "").strip()
        if not content:
            raise ValueError("Conversation turn content cannot be empty.")
        return cls(str(uuid4()), role, content, datetime.now(timezone.utc))

@dataclass(frozen=True, slots=True)
class ConversationMemorySnapshot:
    session_id: str
    turns: tuple[ConversationTurn, ...]
    summary: str
    last_referenced_entity: str | None
    turn_count: int

@dataclass(slots=True)
class ConversationMemory:
    session_id: str
    max_turns: int = 10
    max_context_characters: int = 6000
    _turns: list[ConversationTurn] = field(default_factory=list, repr=False)
    _summary: str = field(default="", repr=False)
    _last_referenced_entity: str | None = field(default=None, repr=False)

    def __post_init__(self):
        self.session_id = (self.session_id or "").strip()
        if not self.session_id: raise ValueError("session_id cannot be empty.")
        if self.max_turns <= 0: raise ValueError("max_turns must be greater than zero.")
        if self.max_context_characters <= 0: raise ValueError("max_context_characters must be greater than zero.")

    @property
    def turns(self): return tuple(self._turns)
    @property
    def summary(self): return self._summary
    @property
    def last_referenced_entity(self): return self._last_referenced_entity
    @property
    def turn_count(self): return len(self._turns)
    @property
    def is_empty(self): return not self._turns and not self._summary and not self._last_referenced_entity

    def add_user_message(self, content): return self.add_turn(ConversationRole.USER, content)
    def add_assistant_message(self, content): return self.add_turn(ConversationRole.ASSISTANT, content)

    def add_turn(self, role, content):
        turn = ConversationTurn.create(role, content)
        self._turns.append(turn)
        self._trim_history()
        return turn

    def add_turns(self, turns: Iterable[ConversationTurn]):
        for turn in turns:
            if not isinstance(turn, ConversationTurn):
                raise TypeError("turns must contain ConversationTurn objects.")
            self._turns.append(turn)
        self._trim_history()

    def oldest_turns(self, count: int):
        return tuple(self._turns[:max(0, count)])

    def remove_oldest_turns(self, count: int):
        count = min(max(0, count), len(self._turns))
        removed = tuple(self._turns[:count])
        del self._turns[:count]
        return removed

    def set_last_referenced_entity(self, entity):
        if entity is None:
            self._last_referenced_entity = None
        else:
            entity = entity.strip()
            self._last_referenced_entity = entity or None

    def set_summary(self, summary): self._summary = (summary or "").strip()

    def append_summary(self, text):
        text = (text or "").strip()
        if text:
            self._summary = f"{self._summary}\n{text}".strip()

    def render_recent_history(self, max_turns=None):
        max_turns = self.max_turns if max_turns is None else max_turns
        if max_turns <= 0: return ""
        lines = [f"{'User' if t.role == ConversationRole.USER else 'Assistant'}: {t.content}"
                 for t in self._turns[-max_turns:]]
        value = "\n".join(lines)
        return value[-self.max_context_characters:]

    def build_context(self):
        parts = []
        if self._summary: parts.append(f"Conversation Summary:\n{self._summary}")
        if self._last_referenced_entity: parts.append(f"Last Referenced Entity:\n{self._last_referenced_entity}")
        history = self.render_recent_history()
        if history: parts.append(f"Recent Conversation:\n{history}")
        return "\n\n".join(parts)[-self.max_context_characters:]

    def snapshot(self):
        return ConversationMemorySnapshot(self.session_id, tuple(self._turns), self._summary,
                                          self._last_referenced_entity, len(self._turns))

    def clear(self):
        self._turns.clear()
        self._summary = ""
        self._last_referenced_entity = None

    def _trim_history(self):
        overflow = len(self._turns) - self.max_turns
        if overflow > 0:
            del self._turns[:overflow]
