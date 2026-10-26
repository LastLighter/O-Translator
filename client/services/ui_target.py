"""Geometry and selection model for auto-detected windows and UI controls.

The native probes in :mod:`client.services.win32_targets` and
:mod:`client.services.uia_targets` work in physical virtual-desktop pixels,
because that is what Win32 and UI Automation report for a per-monitor DPI-aware
process.  This module keeps that coordinate space platform-neutral and provides
the ordering rules used by the overlay:

* small rectangles (inner controls) come first;
* exact duplicates from different APIs are collapsed;
* the screen itself is always available as the outermost fallback.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

SCREEN = "screen"
WINDOW = "window"
CLIENT = "client"
CONTROL = "control"
ELEMENT = "element"

KIND_LABELS = {
    SCREEN: "整个屏幕",
    WINDOW: "窗口",
    CLIENT: "客户区",
    CONTROL: "控件",
    ELEMENT: "界面元素",
}

# Lower value wins when two probes report the same rectangle.
_KIND_PRIORITY = {ELEMENT: 0, CONTROL: 1, CLIENT: 2, WINDOW: 3, SCREEN: 4}


@dataclass(frozen=True, slots=True)
class PhysicalRect:
    """A rectangle in physical virtual-desktop pixels."""

    left: int
    top: int
    width: int
    height: int

    @classmethod
    def from_ltrb(cls, left: int, top: int, right: int, bottom: int) -> PhysicalRect:
        return cls(
            int(left),
            int(top),
            max(0, int(right) - int(left)),
            max(0, int(bottom) - int(top)),
        )

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def intersection(self, other: PhysicalRect) -> PhysicalRect:
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)
        return PhysicalRect.from_ltrb(left, top, right, bottom)

    def __bool__(self) -> bool:
        return not self.is_empty


@dataclass(frozen=True, slots=True)
class UITarget:
    """One selectable layer below the cursor."""

    rect: PhysicalRect
    label: str = ""
    kind: str = WINDOW
    source: str = "win32"
    hwnd: int = 0
    depth: int = 0

    @property
    def area(self) -> int:
        return self.rect.area

    @property
    def kind_label(self) -> str:
        return KIND_LABELS.get(self.kind, KIND_LABELS[ELEMENT])

    @property
    def display_label(self) -> str:
        label = " ".join(self.label.split())
        if label:
            return f"{self.kind_label} · {label}"
        return self.kind_label


@dataclass(frozen=True, slots=True)
class WindowScan:
    """Fast Win32 pass result, also used to seed the slower UIA pass."""

    targets: tuple[UITarget, ...] = ()
    top_hwnd: int = 0
    pid: int = 0
    app: str = ""
    top_class: str = ""
    deep_hwnd: int = 0
    deep_class: str = ""


def screen_target(rect: PhysicalRect, label: str) -> UITarget:
    return UITarget(
        rect=rect,
        label=label,
        kind=SCREEN,
        source="screen",
        depth=1_000_000,
    )


def _dedupe_key(rect: PhysicalRect) -> tuple[int, int, int, int]:
    return (rect.left, rect.top, rect.width, rect.height)


def _target_score(target: UITarget) -> tuple[int, int, int]:
    """Prefer labelled, specific descriptions when rectangles are identical."""
    return (
        1 if target.label.strip() else 0,
        -_KIND_PRIORITY.get(target.kind, _KIND_PRIORITY[ELEMENT]),
        1 if target.source == "uia" else 0,
    )


def normalize_targets(
    targets: Iterable[UITarget],
    *,
    screen: PhysicalRect,
    screen_label: str = "屏幕",
    limit: int = 14,
    min_side: int = 4,
) -> list[UITarget]:
    """Clip, filter, deduplicate, and order targets from innermost to outermost.

    Tiny slivers (splitter grips, scrollbar arrows, one-pixel container hints)
    are dropped because they are hard to capture and almost never what the user
    wants.  Every rectangle is clipped to the screen under the cursor so mixed
    DPI and multi-monitor windows cannot paint outside their pane.
    """
    visible: list[UITarget] = []
    seen: dict[tuple[int, int, int, int], int] = {}

    for target in targets:
        clipped = target.rect.intersection(screen)
        if clipped.width < min_side or clipped.height < min_side:
            continue
        candidate = UITarget(
            rect=clipped,
            label=target.label,
            kind=target.kind,
            source=target.source,
            hwnd=target.hwnd,
            depth=target.depth,
        )
        key = _dedupe_key(clipped)
        previous_index = seen.get(key)
        if previous_index is None:
            seen[key] = len(visible)
            visible.append(candidate)
            continue
        previous = visible[previous_index]
        if _target_score(candidate) > _target_score(previous):
            visible[previous_index] = candidate

    # Smallest area first = most specific element under the cursor.  Depth is a
    # tie-breaker for identical-sized nested containers.
    visible.sort(
        key=lambda item: (
            item.area,
            _KIND_PRIORITY.get(item.kind, 99),
            -item.depth,
        )
    )
    result = visible[: max(1, limit)]
    if screen.area > 0 and not any(
        item.kind == SCREEN or item.rect == screen for item in result
    ):
        result.append(screen_target(screen, screen_label))
    return result


class TargetStack:
    """Selection state for the highlighted rectangle."""

    def __init__(self) -> None:
        self._targets: list[UITarget] = []
        self._index = 0

    @property
    def targets(self) -> list[UITarget]:
        return list(self._targets)

    @property
    def index(self) -> int:
        return self._index

    @property
    def current(self) -> UITarget | None:
        if 0 <= self._index < len(self._targets):
            return self._targets[self._index]
        return None

    @property
    def count(self) -> int:
        return len(self._targets)

    def update(self, targets: Iterable[UITarget], *, preserve: bool = False) -> None:
        """Replace the target list and choose the highlighted layer.

        Index 0 is the default (innermost) layer, so a fresh detection always
        re-selects it.  ``preserve`` keeps an explicitly chosen outer layer
        (wheel/Tab) by rectangle, but it must not pin the default layer:
        otherwise the coarse Win32 window stays highlighted after UI Automation
        reports the actual control underneath the pointer.
        """
        previous = self.current
        was_default = self._index == 0
        self._targets = list(targets)
        if not self._targets:
            self._index = 0
            return
        if preserve and previous is not None and not was_default:
            for index, target in enumerate(self._targets):
                if target.rect == previous.rect:
                    self._index = index
                    return
        self._index = 0

    def cycle(self, step: int) -> UITarget | None:
        if not self._targets:
            return None
        self._index = (self._index + int(step)) % len(self._targets)
        return self.current

    def select(self, index: int) -> UITarget | None:
        if not self._targets:
            self._index = 0
            return None
        self._index = max(0, min(len(self._targets) - 1, int(index)))
        return self.current