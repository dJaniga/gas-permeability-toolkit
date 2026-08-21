"""Running an acquisition loop and a live display at the same time.

Two facts about this program point in opposite directions:

* matplotlib's GUI backends are **not thread-safe**. A figure must be created
  and drawn on the thread that owns the event loop, which is the main thread.
* the acquisition loop **must not be interrupted by a draw**. A redraw of a
  five-panel figure costs on the order of 0.15 s against a 0.1 s sample slot at
  the default 10 Hz, and drawing it inside the per-sample callback spends a
  third of the run not sampling.

The only arrangement that satisfies both is the one that looks backwards: the
**display keeps the main thread** and the **acquisition moves to a worker**.
Putting the plot on a worker thread instead -- the arrangement most people
reach for first -- is what is actually forbidden.

What that buys is not free. The draw is largely Python-level and holds the GIL,
so a worker doing Python work still contends with it; what saves the loop is
that it spends nearly all of its slot asleep, and both ``time.sleep`` and the
DAQ read release the GIL. The residual is thread-switch jitter of a few
milliseconds against a 100 ms slot, rather than a 150 ms stall every frame.

Nothing here imports matplotlib, the acquisition module, or the config. The
loop and the display are duck-typed on the small protocols below, which is what
makes the threading testable with fakes rather than with a rig and a window.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

__all__ = ["AcquisitionLike", "DisplayLike", "run_with_display", "DEFAULT_POLL_INTERVAL_S"]

#: How often the main thread services the display while the worker samples.
#: Short enough that the window stays responsive to a drag or a close box --
#: a GUI that only pumps events every redraw is reported as "not responding"
#: by the window manager -- and long enough not to spin a core.
DEFAULT_POLL_INTERVAL_S = 0.02


class AcquisitionLike(Protocol):
    """The part of an acquisition loop this driver needs."""

    def run(self, *, install_signal_handler: bool = True) -> Any:
        """Sample until stopped. Called on the worker thread."""

    def request_stop(self, reason: str = ...) -> None:
        """Ask the loop to finish after the current sample. Called from another thread."""


class DisplayLike(Protocol):
    """The part of a live plot this driver needs. All called on the main thread."""

    def maybe_redraw(self, now: float | None = ..., *, force: bool = ...) -> bool:
        """Redraw if due (or if forced)."""

    def pump(self) -> None:
        """Service the GUI's event queue without redrawing."""


def run_with_display(
    loop: AcquisitionLike,
    display: DisplayLike,
    *,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
    install_signal_handler: bool = True,
) -> None:
    """Run ``loop`` on a worker thread while ``display`` refreshes on this one.

    Returns when the loop stops. Anything the loop raised is re-raised here, on
    the caller's thread, so a ``DaqError`` reaches the command that knows how to
    report it rather than dying unnoticed inside a thread.

    Ctrl+C is handled here rather than in the loop, because ``signal.signal``
    only works on the main thread and the loop is no longer on it. The first
    one asks the loop to stop after its current sample, which is what makes a
    partially-collected run still get written; a second one gives up waiting and
    raises, so a wedged DAQ read cannot trap the operator in an
    uninterruptible process.

    Args:
        loop: The acquisition loop. Its ``run`` is called with
            ``install_signal_handler=False`` -- it is not on the main thread and
            could not install one anyway.
        display: Refreshed between polls. Its failures are the plot's own
            business and never reach the loop.
        poll_interval_s: How long the main thread sleeps between services.
        install_signal_handler: Set false when the caller owns SIGINT already.
    """
    failure: list[BaseException] = []
    interrupts = 0

    def worker() -> None:
        try:
            loop.run(install_signal_handler=False)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            failure.append(exc)

    def handler(signum, frame):  # noqa: ANN001, ARG001
        nonlocal interrupts
        interrupts += 1
        if interrupts == 1:
            logger.info("Stop requested; finishing the current sample.")
            loop.request_stop("interrupted")
        else:
            logger.warning("Second interrupt; abandoning the wait.")

    previous_handler = None
    if install_signal_handler:
        try:
            previous_handler = signal.signal(signal.SIGINT, handler)
        except ValueError:  # pragma: no cover - not on the main thread
            previous_handler = None

    thread = threading.Thread(target=worker, name="gasperm-acquisition", daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            _safely(display.maybe_redraw, "redraw")
            _safely(display.pump, "event pump")
            if interrupts > 1:
                raise KeyboardInterrupt
            sleep(poll_interval_s)
        # The worker is done; nothing else can be waiting on it.
        thread.join()
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)

    # A last frame, unconditionally: the run ended somewhere between two ticks,
    # and the operator should be looking at where it finished rather than at
    # whatever the last interval happened to catch.
    _safely(lambda: display.maybe_redraw(force=True), "final redraw")

    if failure:
        raise failure[0]


def _safely(action: Callable[[], Any], what: str) -> None:
    """Run a display action, logging rather than propagating its failure.

    The display must never be able to end a run: a window closed mid-run is an
    operator changing their mind about watching, not a reason to stop measuring.
    """
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - the window may have gone away
        logger.warning("Live display %s failed: %s", what, exc)
