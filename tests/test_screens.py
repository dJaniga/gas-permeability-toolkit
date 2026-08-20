"""Putting a plot window on a chosen monitor.

The window calls are exercised against a recording double rather than a real
toolkit: the decision logic and the *order* of the toolkit calls are what can be
got wrong, and both are testable with no desktop at all. Enumeration itself is
an OS query, so only its contract is asserted here.
"""

from __future__ import annotations

import pytest

from gasperm.screens import (
    Screen,
    choose_screen,
    describe_screens,
    list_screens,
    place_window,
)


def screen(index: int, x: int = 0, *, primary: bool = False, taskbar: int = 40) -> Screen:
    return Screen(
        index=index, x=x, y=0, width=1920, height=1080,
        work_x=x, work_y=0, work_width=1920, work_height=1080 - taskbar,
        primary=primary,
    )


ONE = [screen(1, 0, primary=True)]
TWO = [screen(1, 0, primary=True), screen(2, 1920)]


class FakeTkWindow:
    """Records what a Tk window would have been asked to do, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def wm_geometry(self, spec: str) -> None:
        self.calls.append(("geometry", spec))

    def update_idletasks(self) -> None:
        self.calls.append(("update",))

    def state(self, value: str) -> None:
        self.calls.append(("state", value))

    def attributes(self, name: str, value) -> None:  # noqa: ANN001
        self.calls.append(("attributes", name, value))


class FakeQtWindow:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def setGeometry(self, x, y, w, h) -> None:  # noqa: N802, ANN001
        self.calls.append(("setGeometry", x, y, w, h))

    def showMaximized(self) -> None:  # noqa: N802
        self.calls.append(("showMaximized",))

    def showFullScreen(self) -> None:  # noqa: N802
        self.calls.append(("showFullScreen",))


class TestChooseScreen:
    def test_asking_for_nothing_places_nothing(self):
        chosen, complaint = choose_screen(TWO, None)
        assert chosen is None and complaint == ""

    def test_the_second_screen_is_the_second_one(self):
        chosen, complaint = choose_screen(TWO, 2)
        assert chosen is TWO[1]
        assert complaint == ""

    def test_a_missing_screen_falls_back_to_the_primary_and_says_so(self):
        """A plot quietly opening on the wrong monitor for hours is exactly the
        sort of thing nobody reports and everybody works around."""
        chosen, complaint = choose_screen(ONE, 2)
        assert chosen is ONE[0]
        assert "reports 1 screen" in complaint
        assert "plot.monitor" in complaint

    def test_the_fallback_prefers_the_primary_over_the_first(self):
        screens = [screen(1, 0), screen(2, 1920, primary=True)]
        chosen, _ = choose_screen(screens, 5)
        assert chosen.primary is True

    def test_an_unreadable_layout_places_nothing_but_complains(self):
        chosen, complaint = choose_screen([], 2)
        assert chosen is None
        assert "could not be read" in complaint

    def test_a_nonsense_index_is_refused(self):
        chosen, complaint = choose_screen(TWO, 0)
        assert chosen is None and "1 or more" in complaint

    def test_screens_describe_themselves_for_the_message(self):
        assert "1920x1080" in describe_screens(TWO)
        assert "(primary)" in describe_screens(TWO)


class TestPlaceWindowTk:
    def test_nothing_asked_for_does_nothing(self):
        window = FakeTkWindow()
        assert place_window(window, None, "normal") == ""
        assert window.calls == []

    def test_a_move_uses_the_work_area(self):
        """A maximised window should not sit under the taskbar."""
        window = FakeTkWindow()
        place_window(window, TWO[1], "normal")
        assert window.calls[0] == ("geometry", "1920x1040+1920+0")

    def test_fullscreen_covers_the_whole_monitor(self):
        """Including the taskbar strip, unlike maximised."""
        window = FakeTkWindow()
        place_window(window, TWO[1], "fullscreen")
        assert window.calls[0] == ("geometry", "1920x1080+1920+0")
        assert ("attributes", "-fullscreen", True) in window.calls

    def test_the_move_happens_before_the_maximise(self):
        """Tk maximises onto whichever monitor the window is already on, so
        zooming first would fill the wrong screen."""
        window = FakeTkWindow()
        place_window(window, TWO[1], "maximised")
        kinds = [c[0] for c in window.calls]
        assert kinds.index("geometry") < kinds.index("state")
        assert ("state", "zoomed") in window.calls

    def test_it_settles_the_geometry_before_zooming(self):
        window = FakeTkWindow()
        place_window(window, TWO[1], "maximised")
        kinds = [c[0] for c in window.calls]
        assert kinds.index("update") < kinds.index("state")

    def test_a_toolkit_that_cannot_zoom_keeps_the_explicit_geometry(self):
        """On a platform without 'zoomed', the work-area geometry already fills
        the screen -- so the failure is swallowed rather than losing the move."""

        class NoZoom(FakeTkWindow):
            def state(self, value):  # noqa: ANN001
                raise RuntimeError("bad window state name")

        window = NoZoom()
        assert place_window(window, TWO[1], "maximised") != ""
        assert window.calls[0][0] == "geometry"

    def test_a_window_that_raises_does_not_propagate(self):
        """Placement is additive to a plot, which is additive to a run."""

        class Broken(FakeTkWindow):
            def wm_geometry(self, spec):  # noqa: ANN001
                raise RuntimeError("display gone")

        assert place_window(Broken(), TWO[1], "fullscreen") == ""


class TestPlaceWindowQt:
    def test_geometry_then_maximise(self):
        window = FakeQtWindow()
        place_window(window, TWO[1], "maximised")
        assert window.calls == [
            ("setGeometry", 1920, 0, 1920, 1040),
            ("showMaximized",),
        ]

    def test_fullscreen_uses_the_monitor_rect(self):
        window = FakeQtWindow()
        place_window(window, TWO[1], "fullscreen")
        assert window.calls[0] == ("setGeometry", 1920, 0, 1920, 1080)
        assert ("showFullScreen",) in window.calls


class TestPlaceWindowOther:
    def test_an_unknown_toolkit_is_left_alone(self):
        assert place_window(object(), TWO[1], "fullscreen") == ""

    def test_no_window_at_all_is_fine(self):
        assert place_window(None, TWO[1], "fullscreen") == ""


class TestEnumeration:
    def test_it_returns_screens_or_nothing_but_never_raises(self):
        """The contract callers rely on; the values depend on the machine."""
        found = list_screens()
        assert isinstance(found, list)
        for item in found:
            assert isinstance(item, Screen)
            assert item.width > 0 and item.height > 0
            assert item.work_width > 0 and item.work_height > 0

    def test_indices_are_one_based_and_contiguous(self):
        found = list_screens()
        assert [s.index for s in found] == list(range(1, len(found) + 1))

    def test_at_most_one_primary(self):
        found = list_screens()
        assert sum(1 for s in found if s.primary) <= 1


class TestPlotIntegration:
    """The figure-level entry point, which reads the config."""

    def config(self, **plot):
        from gasperm.config import GaspermConfig

        config = GaspermConfig()
        for key, value in plot.items():
            setattr(config.run.plot, key, value)
        return config

    def figure_with(self, window):
        class Manager:
            pass

        class Canvas:
            pass

        class Figure:
            pass

        manager, canvas, figure = Manager(), Canvas(), Figure()
        manager.window = window
        canvas.manager = manager
        figure.canvas = canvas
        return figure

    def test_the_default_config_places_nothing(self):
        from gasperm.plotting import place_figure

        window = FakeTkWindow()
        assert place_figure(self.figure_with(window), self.config().run.plot) == ""
        assert window.calls == []

    def test_a_figure_without_a_window_is_tolerated(self):
        """Agg has a manager with no window, and saving a PNG must still work."""
        from gasperm.plotting import place_figure

        class Bare:
            pass

        figure, canvas = Bare(), Bare()
        canvas.manager = Bare()
        figure.canvas = canvas
        assert place_figure(figure, self.config(monitor=2).run.plot) == ""

    def test_no_config_places_nothing(self):
        from gasperm.plotting import place_figure

        assert place_figure(self.figure_with(FakeTkWindow()), None) == ""

    def test_a_requested_mode_reaches_the_window(self, monkeypatch):
        from gasperm import screens as screens_module
        from gasperm.plotting import place_figure

        monkeypatch.setattr(screens_module, "list_screens", lambda: TWO)
        window = FakeTkWindow()
        result = place_figure(
            self.figure_with(window), self.config(monitor=2, window="fullscreen").run.plot
        )
        assert "screen 2" in result
        assert window.calls[0] == ("geometry", "1920x1080+1920+0")
        assert ("attributes", "-fullscreen", True) in window.calls

    def test_a_missing_monitor_warns_and_still_opens(self, monkeypatch, caplog):
        from gasperm import screens as screens_module
        from gasperm.plotting import place_figure

        monkeypatch.setattr(screens_module, "list_screens", lambda: ONE)
        window = FakeTkWindow()
        with caplog.at_level("WARNING"):
            place_figure(
                self.figure_with(window),
                self.config(monitor=2, window="fullscreen").run.plot,
            )
        assert "reports 1 screen" in caplog.text
        # It still opened, on the screen that exists.
        assert window.calls[0] == ("geometry", "1920x1080+0+0")


class TestConfig:
    def test_the_defaults_change_nothing(self):
        from gasperm.config import GaspermConfig

        plot = GaspermConfig().run.plot
        assert plot.monitor is None
        assert plot.window == "normal"

    def test_either_spelling_of_maximised(self):
        from gasperm.config import LivePlotConfig

        assert LivePlotConfig(window="maximized").window == "maximised"
        assert LivePlotConfig(window="maximised").window == "maximised"

    def test_a_zero_monitor_is_refused(self):
        from pydantic import ValidationError

        from gasperm.config import LivePlotConfig

        with pytest.raises(ValidationError):
            LivePlotConfig(monitor=0)

    def test_an_unknown_window_mode_is_refused(self):
        from pydantic import ValidationError

        from gasperm.config import LivePlotConfig

        with pytest.raises(ValidationError):
            LivePlotConfig(window="enormous")
