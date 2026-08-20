"""Putting a plot window on a chosen monitor.

A rig bench usually has two screens: the console on one, and the live plot left
running on the other for hours. Matplotlib has no concept of a second monitor,
so this module supplies one.

Everything here is **best-effort and never fatal**. A plot is additive to a run
(see :mod:`gasperm.plotting`), and window placement is additive to the plot: if
the monitor is unplugged, the backend does not support geometry, or the OS query
fails, the window opens wherever it would have anyway and the run carries on.
The only thing that is never silent is asking for a screen that is not there,
because that is a configuration mistake rather than an environment limit.

Two pieces, separated because only one of them can be tested without a desktop:

* :func:`list_screens` asks the OS what monitors exist. Windows-only for now,
  via ``EnumDisplayMonitors`` -- Tk, which is the backend matplotlib picks here,
  cannot enumerate monitors at all and reports only a single merged desktop.
* :func:`place_window` decides and applies geometry given that list. It takes a
  window object rather than importing any GUI toolkit, so it is exercised
  against a recording double in the tests.

**One caveat, on mixed-DPI desktops.** A DPI-unaware process -- which is what
Python with Tk normally is -- sees *virtualised* coordinates from both the OS
query and the toolkit. That is harmless while every monitor runs at the same
scaling, because the two agree with each other and placement is self-consistent.
It is not harmless when a 150 %-scaled laptop panel sits beside a 100 % external
screen: the virtual rectangles no longer match the physical ones and the window
can land short. Making the process DPI-aware would fix the coordinates and shrink
every plot's text, so it is deliberately not done here; if placement misses on
such a desktop, setting the monitors to matching scale is the cheaper fix.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Literal, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "Screen",
    "WindowMode",
    "choose_screen",
    "describe_screens",
    "list_screens",
    "place_window",
]

#: How much of the chosen screen the window should take.
WindowMode = Literal["normal", "maximised", "fullscreen"]


@dataclass(frozen=True)
class Screen:
    """One monitor, in the OS's virtual-desktop coordinates."""

    #: 1-based, in the order the OS enumerates them. The primary screen is not
    #: necessarily first, which is why :attr:`primary` is carried separately.
    index: int
    x: int
    y: int
    width: int
    height: int
    #: The area excluding the taskbar. A maximised window gets this; a
    #: fullscreen one covers the monitor rect instead.
    work_x: int
    work_y: int
    work_width: int
    work_height: int
    primary: bool = False

    def geometry(self, *, work_area: bool) -> tuple[int, int, int, int]:
        """``(x, y, width, height)`` for the requested area."""
        if work_area:
            return (self.work_x, self.work_y, self.work_width, self.work_height)
        return (self.x, self.y, self.width, self.height)

    def __str__(self) -> str:
        tag = " (primary)" if self.primary else ""
        return f"screen {self.index}: {self.width}x{self.height} at ({self.x},{self.y}){tag}"


def list_screens() -> list[Screen]:
    """Every monitor the OS reports, left to right as it enumerates them.

    Returns an empty list on a platform this does not know how to ask, or when
    the query fails -- callers treat that as "place nothing", not as an error.
    """
    if not sys.platform.startswith("win"):
        # Nothing else is implemented yet. X11/Wayland/macOS each need their own
        # query, and guessing from a merged desktop size would put the window in
        # the wrong place rather than leaving it alone.
        logger.debug("Screen enumeration is not implemented on %s.", sys.platform)
        return []
    try:
        return _windows_screens()
    except Exception as exc:  # noqa: BLE001 - a display query must never raise
        logger.debug("Could not enumerate monitors: %s", exc)
        return []


def _windows_screens() -> list[Screen]:
    """Monitors via ``user32.EnumDisplayMonitors``."""
    import ctypes
    from ctypes import wintypes

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    MONITORINFOF_PRIMARY = 0x1
    user32 = ctypes.windll.user32
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )

    found: list[Screen] = []

    def collect(handle, _hdc, _rect, _param):  # noqa: ANN001
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            return 1
        monitor, work = info.rcMonitor, info.rcWork
        found.append(
            Screen(
                index=len(found) + 1,
                x=monitor.left,
                y=monitor.top,
                width=monitor.right - monitor.left,
                height=monitor.bottom - monitor.top,
                work_x=work.left,
                work_y=work.top,
                work_width=work.right - work.left,
                work_height=work.bottom - work.top,
                primary=bool(info.dwFlags & MONITORINFOF_PRIMARY),
            )
        )
        return 1

    user32.EnumDisplayMonitors(0, None, callback_type(collect), 0)
    return found


def describe_screens(screens: Sequence[Screen]) -> str:
    """One line naming what was found, for a warning message."""
    if not screens:
        return "no screens reported"
    return "; ".join(str(screen) for screen in screens)


def choose_screen(
    screens: Sequence[Screen], requested: int | None
) -> tuple[Screen | None, str]:
    """Pick the requested monitor, and say what happened.

    Args:
        screens: What :func:`list_screens` found.
        requested: 1-based monitor number, or ``None`` to leave placement alone.

    Returns:
        ``(screen, complaint)``. ``screen`` is ``None`` when nothing should be
        moved. ``complaint`` is empty unless the operator asked for something
        this machine cannot give, in which case it is worth showing: a plot
        quietly opening on the wrong monitor for hours is exactly the sort of
        thing nobody reports and everybody works around.
    """
    if requested is None:
        return None, ""
    if not screens:
        return None, (
            "plot.monitor is set, but the monitor layout could not be read on this "
            "platform, so the window is left where the desktop puts it."
        )
    if requested < 1:
        return None, f"plot.monitor must be 1 or more, got {requested}."
    if requested > len(screens):
        fallback = next((s for s in screens if s.primary), screens[0])
        return fallback, (
            f"plot.monitor is {requested} but this machine reports "
            f"{len(screens)} screen(s) ({describe_screens(screens)}). Using "
            f"screen {fallback.index} instead -- plug the second monitor in, or "
            "set plot.monitor: null."
        )
    return screens[requested - 1], ""


def place_window(window: Any, screen: Screen | None, mode: WindowMode = "normal") -> str:
    """Move ``window`` onto ``screen`` and size it, returning what was done.

    ``window`` is the backend's native window -- ``manager.window`` -- and is
    identified by what it can do rather than by importing a GUI toolkit, so this
    neither drags Qt into a Tk install nor needs a display to be unit-tested.

    **The move has to happen before the maximise.** Both toolkits maximise onto
    whichever monitor the window currently occupies, so zooming first would fill
    the wrong screen and the subsequent move would be ignored or would un-zoom
    it.

    Returns:
        A short description for the log, or ``""`` when nothing was done.
    """
    if window is None:
        return ""
    if screen is None and mode == "normal":
        return ""

    fullscreen = mode == "fullscreen"
    if screen is not None:
        x, y, width, height = screen.geometry(work_area=not fullscreen)
    else:
        x = y = width = height = None

    try:
        if hasattr(window, "wm_geometry"):  # Tk
            return _place_tk(window, x, y, width, height, mode, screen)
        if hasattr(window, "setGeometry"):  # Qt
            return _place_qt(window, x, y, width, height, mode, screen)
    except Exception as exc:  # noqa: BLE001 - placement must never kill a plot
        logger.debug("Could not place the plot window: %s", exc)
        return ""

    logger.debug(
        "Plot window placement is not supported for %s.", type(window).__name__
    )
    return ""


def _place_tk(window, x, y, width, height, mode, screen) -> str:  # noqa: ANN001
    if x is not None:
        # Size and position together, so the window lands on the target monitor
        # before anything asks it to fill one.
        window.wm_geometry(f"{width}x{height}+{x}+{y}")
        window.update_idletasks()
    if mode == "fullscreen":
        window.attributes("-fullscreen", True)
    elif mode == "maximised":
        # 'zoomed' is the Windows/Tk spelling of maximise; elsewhere it raises
        # and the explicit geometry set above already covers the work area.
        try:
            window.state("zoomed")
        except Exception:  # noqa: BLE001
            pass
    return _describe(mode, screen)


def _place_qt(window, x, y, width, height, mode, screen) -> str:  # noqa: ANN001
    if x is not None:
        window.setGeometry(x, y, width, height)
    if mode == "fullscreen":
        window.showFullScreen()
    elif mode == "maximised":
        window.showMaximized()
    return _describe(mode, screen)


def _describe(mode: WindowMode, screen: Screen | None) -> str:
    where = f"screen {screen.index}" if screen is not None else "the current screen"
    if mode == "normal":
        return f"moved to {where}"
    return f"{mode} on {where}"
