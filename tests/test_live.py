"""``gasperm.live``: acquisition on a worker, drawing on the main thread.

The arrangement is the one that looks backwards -- the *display* keeps the main
thread, because matplotlib's GUI backends may only draw there, and the
*acquisition* moves off it. These drive the driver with fakes rather than with a
rig and a window, which is what the duck-typed protocols in ``live.py`` are for.

Threads make tests flaky when they are asserted on by sleeping. Nothing here
sleeps to wait: the fakes use events and counters, so every assertion is about
something that has definitely happened.
"""

from __future__ import annotations

import threading

import pytest

from gasperm.live import run_with_display


class FakeLoop:
    """An acquisition loop that runs until stopped, or for a fixed count."""

    def __init__(self, samples: int | None = None, *, raises: BaseException | None = None):
        self.samples = samples
        self.raises = raises
        self.ran_on: str | None = None
        self.install_signal_handler: bool | None = None
        self.stopped_for: str | None = None
        self.taken = 0
        self._stop = threading.Event()
        self.started = threading.Event()

    def run(self, *, install_signal_handler: bool = True):
        self.install_signal_handler = install_signal_handler
        self.ran_on = threading.current_thread().name
        self.started.set()
        if self.raises is not None:
            raise self.raises
        while not self._stop.is_set():
            self.taken += 1
            if self.samples is not None and self.taken >= self.samples:
                break
        return self.taken

    def request_stop(self, reason: str = "requested") -> None:
        self.stopped_for = reason
        self._stop.set()


class FakeDisplay:
    """Counts what the driver asked it to do, and on which thread.

    Optionally stops ``loop`` once it has been serviced ``stop_after`` times.
    A fake loop with no I/O in it finishes long before the driver's first poll,
    so a test that wants to observe the two running *together* has to let the
    display be what ends the run rather than a sample count.
    """

    def __init__(self, *, fails: bool = False, loop=None, stop_after: int | None = None):
        self.redraws = 0
        self.forced = 0
        self.pumps = 0
        self.threads: set[str] = set()
        self.fails = fails
        self.loop = loop
        self.stop_after = stop_after

    def _serviced(self) -> None:
        self.threads.add(threading.current_thread().name)
        if (
            self.stop_after is not None
            and self.loop is not None
            and self.pumps >= self.stop_after
        ):
            self.loop.request_stop("display had enough")
        # Counted before any failure, so a broken display is still known to
        # have been reached.
        if self.fails:
            raise RuntimeError("the window is gone")

    def maybe_redraw(self, now=None, *, force=False) -> bool:
        self.redraws += 1
        if force:
            self.forced += 1
        self._serviced()
        return True

    def pump(self) -> None:
        self.pumps += 1
        self._serviced()


def no_sleep(_seconds: float) -> None:
    """The driver's pacing sleep, removed so tests never wait on wall time."""


class TestWhoRunsWhere:
    """The whole point of the module: which thread does which job."""

    def test_the_loop_leaves_the_main_thread(self):
        loop, display = FakeLoop(samples=5), FakeDisplay()
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert loop.ran_on == "gasperm-acquisition"
        assert loop.ran_on != threading.current_thread().name

    def test_the_display_keeps_it(self):
        """matplotlib GUI backends may only draw on the thread that owns them."""
        loop, display = FakeLoop(samples=5), FakeDisplay()
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert display.threads == {threading.current_thread().name}

    def test_the_loop_is_told_not_to_install_a_handler(self):
        """It could not anyway -- signal.signal only works on the main thread."""
        loop, display = FakeLoop(samples=5), FakeDisplay()
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert loop.install_signal_handler is False

    def test_the_worker_is_joined_before_returning(self):
        """A run that returns while its DAQ is still being read is a bug."""
        loop, display = FakeLoop(samples=3), FakeDisplay()
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert not any(
            t.name == "gasperm-acquisition" for t in threading.enumerate()
        )


class TestFinishing:
    def test_the_last_frame_is_drawn_unconditionally(self):
        """The run ended between two ticks; show where it actually finished."""
        loop, display = FakeLoop(samples=3), FakeDisplay()
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert display.forced == 1

    def test_the_display_is_serviced_while_the_worker_runs(self):
        """The two really are concurrent: the display ends a loop still running."""
        loop = FakeLoop()  # runs until stopped
        display = FakeDisplay(loop=loop, stop_after=3)
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert display.pumps >= 3
        assert display.redraws >= 3
        assert loop.stopped_for == "display had enough"
        assert loop.taken > 0

    def test_a_loop_failure_is_re_raised_on_the_calling_thread(self):
        """Otherwise a DaqError dies inside the thread and the run looks clean."""
        from gasperm.hardware.daq import DaqError

        loop = FakeLoop(raises=DaqError("simulated unplug"))
        with pytest.raises(DaqError, match="simulated unplug"):
            run_with_display(loop, FakeDisplay(), sleep=no_sleep, install_signal_handler=False)

    def test_the_final_frame_is_still_drawn_after_a_failure(self):
        """What the rig was doing when it died is the most useful thing on screen."""
        loop = FakeLoop(raises=RuntimeError("boom"))
        display = FakeDisplay()
        with pytest.raises(RuntimeError):
            run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert display.forced == 1


class TestDisplayFailure:
    """A window is a convenience. It must never be able to end a run."""

    def test_a_broken_display_does_not_stop_the_loop(self):
        """A window closed mid-run is someone deciding not to watch, not a fault."""
        loop, display = FakeLoop(samples=5), FakeDisplay(fails=True)
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert loop.taken == 5
        assert loop.stopped_for is None

    def test_a_display_failing_mid_run_is_swallowed(self):
        """And it is reached *during* the run, not only on the final frame."""
        loop = FakeLoop()
        display = FakeDisplay(fails=True, loop=loop, stop_after=3)
        run_with_display(loop, display, sleep=no_sleep, install_signal_handler=False)
        assert display.pumps >= 3
        assert loop.taken > 0


class TestInterrupt:
    """Ctrl+C is handled here because the loop is no longer on the main thread."""

    def test_the_first_interrupt_asks_the_loop_to_stop(self):
        """Not a kill: the current sample finishes and the run still gets written."""
        import signal as signal_module

        loop = FakeLoop()  # runs until stopped
        display = FakeDisplay()
        fired = threading.Event()

        def interrupt_once(_seconds: float) -> None:
            if not fired.is_set():
                fired.set()
                # Call the installed handler the way the OS would.
                signal_module.getsignal(signal_module.SIGINT)(signal_module.SIGINT, None)

        run_with_display(loop, display, sleep=interrupt_once)
        assert loop.stopped_for == "interrupted"

    def test_the_previous_handler_is_restored(self):
        import signal as signal_module

        before = signal_module.getsignal(signal_module.SIGINT)
        run_with_display(FakeLoop(samples=3), FakeDisplay(), sleep=no_sleep)
        assert signal_module.getsignal(signal_module.SIGINT) is before

    def test_a_second_interrupt_gives_up_waiting(self):
        """A wedged DAQ read must not trap the operator in the process."""
        import signal as signal_module

        loop = _Unstoppable()
        display = FakeDisplay()

        def interrupt_twice(_seconds: float) -> None:
            handler = signal_module.getsignal(signal_module.SIGINT)
            handler(signal_module.SIGINT, None)
            handler(signal_module.SIGINT, None)

        with pytest.raises(KeyboardInterrupt):
            run_with_display(loop, display, sleep=interrupt_twice)
        loop.release()

    def test_the_handler_is_restored_after_a_second_interrupt(self):
        import signal as signal_module

        before = signal_module.getsignal(signal_module.SIGINT)
        loop = _Unstoppable()

        def interrupt_twice(_seconds: float) -> None:
            handler = signal_module.getsignal(signal_module.SIGINT)
            handler(signal_module.SIGINT, None)
            handler(signal_module.SIGINT, None)

        with pytest.raises(KeyboardInterrupt):
            run_with_display(loop, FakeDisplay(), sleep=interrupt_twice)
        assert signal_module.getsignal(signal_module.SIGINT) is before
        loop.release()


class _Unstoppable(FakeLoop):
    """A loop that ignores request_stop -- a DAQ read that never returns."""

    def __init__(self) -> None:
        super().__init__()
        self._release = threading.Event()

    def run(self, *, install_signal_handler: bool = True):
        self.install_signal_handler = install_signal_handler
        self.started.set()
        self._release.wait(timeout=30.0)
        return 0

    def request_stop(self, reason: str = "requested") -> None:
        self.stopped_for = reason  # noted, and deliberately not acted on

    def release(self) -> None:
        self._release.set()
