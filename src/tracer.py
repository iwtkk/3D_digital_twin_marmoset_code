from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional
import os
import time
from src import plantuml


class Tracer:
    def __init__(
        self,
        enabled: bool = False,
        print_steps: bool = False,
        show_time: bool = True,
        print_all: bool = False,
    ) -> None:
        self.enabled = enabled
        self.print_steps = print_steps
        self.show_time = show_time
        self.print_all = print_all
        self._events: List[Dict[str, Any]] = []
        self._step = 0
        self._start_monotonic: Optional[float] = None
        self._summaries: Dict[str, str] = {}

    def reset(self) -> None:
        self._events.clear()
        self._step = 0
        self._start_monotonic = None

    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def set_summaries(self, summaries: Dict[str, str], *, clear: bool = False) -> None:
        if clear:
            self._summaries = {}
        self._summaries.update(summaries)

    def summary_for(self, name: str) -> Optional[str]:
        return self._summaries.get(name)

    def _next_step(self) -> int:
        self._step += 1
        return self._step

    def _record(self, event: Dict[str, Any]) -> None:
        now_monotonic = time.monotonic()
        if self._start_monotonic is None:
            self._start_monotonic = now_monotonic
        event["ts"] = time.time()
        event["elapsed_ms"] = (now_monotonic - self._start_monotonic) * 1000.0
        self._events.append(event)
        if self.print_steps:
            if self.print_all or event.get("detail"):
                print(self._format_event(event))

    def _format_event(self, event: Dict[str, Any]) -> str:
        prefix = f"TRACE {event['step']:04d}"
        if self.show_time:
            elapsed = event.get("elapsed_ms", 0.0)
            prefix = f"{prefix} +{elapsed:09.3f}ms"
        detail = event.get("detail")
        if detail:
            return f"{prefix} {event['dst']}: {detail}"
        return f"{prefix} {event['src']} -> {event['dst']} {event['action']}"

    def timeline(self) -> str:
        return "\n".join(self._format_event(e) for e in self._events)


class SequenceTracer(Tracer):
    def __init__(
        self,
        enabled: bool = False,
        print_steps: bool = False,
        show_time: bool = True,
        print_all: bool = False,
    ) -> None:
        super().__init__(
            enabled=enabled,
            print_steps=print_steps,
            show_time=show_time,
            print_all=print_all,
        )
        self._stack: List[str] = []

    def reset(self) -> None:
        super().reset()
        self._stack.clear()

    def enter(self, name: str, detail: Optional[str] = None) -> None:
        if not self.enabled:
            return
        src = self._stack[-1] if self._stack else "ROOT"
        dst = name
        summary = detail if detail is not None else self.summary_for(name)
        event = {"step": self._next_step(), "src": src, "dst": dst, "action": "enter"}
        if summary:
            event["detail"] = summary
        self._record(event)
        self._stack.append(name)

    def exit(self, name: str) -> None:
        if not self.enabled:
            return
        if self._stack and self._stack[-1] == name:
            self._stack.pop()
        src = name
        dst = self._stack[-1] if self._stack else "ROOT"
        event = {"step": self._next_step(), "src": src, "dst": dst, "action": "exit"}
        self._record(event)

    def event(self, name: str, detail: str) -> None:
        if not self.enabled:
            return
        src = self._stack[-1] if self._stack else "ROOT"
        dst = name
        event = {
            "step": self._next_step(),
            "src": src,
            "dst": dst,
            "action": "event",
            "detail": detail,
        }
        self._record(event)


_TRACE = SequenceTracer(
    enabled=os.getenv("TESTBLENDER_TRACE", "0") not in ("0", "false", "False"),
    print_steps=os.getenv("TESTBLENDER_TRACE_PRINT", "0") not in ("0", "false", "False"),
    show_time=os.getenv("TESTBLENDER_TRACE_TIME", "1") not in ("0", "false", "False"),
    print_all=os.getenv("TESTBLENDER_TRACE_PRINT_ALL", "0") not in ("0", "false", "False"),
)


def set_trace_enabled(
    enabled: bool,
    *,
    print_steps: Optional[bool] = None,
    show_time: Optional[bool] = None,
    print_all: Optional[bool] = None,
) -> None:
    _TRACE.enabled = enabled
    if print_steps is not None:
        _TRACE.print_steps = print_steps
    if show_time is not None:
        _TRACE.show_time = show_time
    if print_all is not None:
        _TRACE.print_all = print_all


def reset_trace() -> None:
    _TRACE.reset()


def trace_events() -> List[Dict[str, Any]]:
    return _TRACE.events()


def trace_to_plantuml() -> str:
    return plantuml.build_sequence_uml(_TRACE.events())


def trace_timeline() -> str:
    return _TRACE.timeline()


def set_trace_summaries(summaries: Dict[str, str], *, clear: bool = False) -> None:
    _TRACE.set_summaries(summaries, clear=clear)


@contextmanager
def trace_scope(name: str, detail: Optional[str] = None):
    _TRACE.enter(name, detail=detail)
    try:
        yield
    finally:
        _TRACE.exit(name)
