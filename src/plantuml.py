from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import inspect
import sys


def _ensure_module(module: Any) -> Any:
    if isinstance(module, str):
        module_obj = sys.modules.get(module)
        if module_obj is None:
            module_obj = __import__(module, fromlist=["*"])
        return module_obj
    return module


def _module_classes(module: Any) -> List[type]:
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
    ]


def _module_functions(module: Any) -> List[Tuple[str, Any]]:
    return [
        (name, func)
        for name, func in inspect.getmembers(module, inspect.isfunction)
        if func.__module__ == module.__name__
    ]


def _class_method_names(cls: type) -> List[str]:
    names: List[str] = []
    for name, func in inspect.getmembers(cls, inspect.isfunction):
        if func.__qualname__.startswith(f"{cls.__name__}."):
            names.append(name)
    return names


def _symbol_names_for_module(module: Any) -> Set[str]:
    names: Set[str] = set()
    for name, _ in _module_functions(module):
        names.add(name)
    for cls in _module_classes(module):
        names.add(cls.__name__)
        for method in _class_method_names(cls):
            names.add(f"{cls.__name__}.{method}")
    return names


def _filter_events_for_module(
    events: Iterable[Dict[str, Any]],
    module: Any,
) -> List[Dict[str, Any]]:
    names = _symbol_names_for_module(module)
    filtered: List[Dict[str, Any]] = []
    for event in events:
        if event.get("src") in names or event.get("dst") in names:
            filtered.append(event)
    return filtered


def _escape_label(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").replace('"', "'")


def _make_alias(name: str, used: Set[str]) -> str:
    base = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name) or "participant"
    alias = base
    counter = 2
    while alias in used:
        alias = f"{base}_{counter}"
        counter += 1
    used.add(alias)
    return alias


def build_class_uml(module: Any) -> str:
    module = _ensure_module(module)
    lines = ["@startuml", "skinparam classAttributeIconSize 0"]
    classes = _module_classes(module)
    if not classes:
        lines.append('note "no classes found"')
    for cls in classes:
        lines.append(f"class {cls.__name__} {{")
        for method in _class_method_names(cls):
            if method.startswith("_"):
                continue
            lines.append(f"  +{method}()")
        lines.append("}")
    lines.append("@enduml")
    return "\n".join(lines)


def build_sequence_uml(events: Iterable[Dict[str, Any]]) -> str:
    events_list = list(events)
    if not events_list:
        return "\n".join(["@startuml", 'note "no trace events"', "@enduml"])
    participants = sorted({e["src"] for e in events_list} | {e["dst"] for e in events_list})
    aliases: Dict[str, str] = {}
    used: Set[str] = set()
    if "ROOT" in participants:
        aliases["ROOT"] = "ROOT"
        used.add("ROOT")
    for name in participants:
        if name == "ROOT":
            continue
        aliases[name] = _make_alias(name, used)
    lines = ["@startuml"]
    if "ROOT" in participants:
        lines.append("actor ROOT")
    for name in participants:
        if name == "ROOT":
            continue
        lines.append(f'participant "{_escape_label(name)}" as {aliases[name]}')
    for event in events_list:
        src = event["src"]
        dst = event["dst"]
        arrow = "->" if event.get("action") != "exit" else "-->"
        label = event.get("action", "event")
        detail = event.get("detail")
        if detail:
            label = f"{label} {_escape_label(str(detail))}"
        label = _escape_label(label)
        lines.append(f"{aliases[src]} {arrow} {aliases[dst]}: {label}")
    lines.append("@enduml")
    return "\n".join(lines)


def build_activity_uml(events: Iterable[Dict[str, Any]]) -> str:
    actions: List[str] = []
    for event in events:
        if event.get("action") == "enter":
            detail = event.get("detail") or event.get("dst")
            actions.append(_escape_label(str(detail)))
    lines = ["@startuml", "start"]
    if not actions:
        lines.append(":no trace events;")
    else:
        for action in actions:
            lines.append(f":{action};")
    lines.append("stop")
    lines.append("@enduml")
    return "\n".join(lines)


def build_uml_bundle(
    module: Any,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    module = _ensure_module(module)
    if events is None:
        events = []
    filtered = _filter_events_for_module(events, module)
    return {
        "class": build_class_uml(module),
        "sequence": build_sequence_uml(filtered),
        "activity": build_activity_uml(filtered),
    }


def write_uml_files(
    module: Any,
    output_root: Path,
    file_stem: str,
    events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    bundle = build_uml_bundle(module, events=events or [])
    output_dir = Path(output_root) / file_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for kind, text in bundle.items():
        path = output_dir / f"{kind}.pu"
        path.write_text(text, encoding="utf-8")
        paths[kind] = path
    return paths
