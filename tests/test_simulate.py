"""Contract tests for the scenario simulator's renderer (tools/simrender.py):
the synthetic desk it draws must read correctly through the REAL Perceiver,
or the scenario runs (tools/sim_scenarios.py) would be testing the app
against frames no real desk produces. Camera-free: everything runs on the
procedural SyntheticBackground (grid detection's bare-grid path included),
with the project's real ink thresholds (config.yaml 0.025/0.035).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from simrender import Scene, SyntheticBackground  # noqa: E402
from inkwatch.perception import Perceiver, StabilityGate  # noqa: E402

INK_LOW, INK_HIGH = 0.025, 0.035


def _perceiver() -> Perceiver:
    return Perceiver(stability=StabilityGate(stability_frames=10))


def _stable_within(perceiver: Perceiver, baseline, scene: Scene, frames: int = 120):
    """Tick until the scene reads stable (or frames run out). The sensor
    noise occasionally resets the quiet count — the point is that it
    always RECOVERS, like the real camera's calm windows."""
    obs = None
    for i in range(frames):
        obs = perceiver.observe(scene.tick(), baseline)
        if obs.stable:
            return i, obs
    return None, obs


def _calibrated_scene(seed: int = 3):
    scene = Scene(SyntheticBackground(), seed=seed)
    perceiver = _perceiver()
    i, obs = _stable_within(perceiver, None, scene)
    assert i is not None, "blank board never stabilized"
    return scene, perceiver, list(obs.ratios)


def test_a_blank_board_stabilizes_with_no_ink_in_any_cell():
    scene = Scene(SyntheticBackground(), seed=3)
    perceiver = _perceiver()
    i, obs = _stable_within(perceiver, None, scene)
    assert i is not None
    assert obs.found
    assert max(obs.ratios) < INK_LOW


def test_a_rendered_x_reads_marked_only_in_its_cell():
    scene, perceiver, baseline = _calibrated_scene()
    scene.draw(6, "X")
    while scene.busy:
        perceiver.observe(scene.tick(), baseline)
    i, obs = _stable_within(perceiver, baseline, scene)
    assert i is not None, "scene never re-stabilized after the draw"
    assert obs.cell_marks[6] == "marked"
    assert all(mark == "none" for j, mark in enumerate(obs.cell_marks) if j != 6)


def test_a_rendered_o_reads_marked():
    scene, perceiver, baseline = _calibrated_scene()
    scene.draw(4, "O")
    while scene.busy:
        perceiver.observe(scene.tick(), baseline)
    i, obs = _stable_within(perceiver, baseline, scene)
    assert i is not None
    assert obs.cell_marks[4] == "marked"


def test_a_half_drawn_mark_is_not_marked_until_finished():
    scene, perceiver, baseline = _calibrated_scene()
    scene.half_draw_then_finish(4, "X", pause_s=1.0)
    mid_reads = []
    while scene.busy and scene._action.data.get("frac", 1.0) < 0.9:
        obs = perceiver.observe(scene.tick(), baseline)
        if obs.stable and obs.cell_marks:
            mid_reads.append(obs.cell_marks[4])
    # A pen lifted mid-mark must never look like a finished one (D6's
    # whole reason to exist).
    assert "marked" not in mid_reads
    while scene.busy:
        perceiver.observe(scene.tick(), baseline)
    i, obs = _stable_within(perceiver, baseline, scene)
    assert i is not None
    assert obs.cell_marks[4] == "marked"


def test_a_scribble_over_a_marked_cell_is_a_clear_change():
    scene, perceiver, baseline = _calibrated_scene()
    scene.draw(4, "X")
    while scene.busy:
        perceiver.observe(scene.tick(), baseline)
    _, obs = _stable_within(perceiver, baseline, scene)
    committed = list(obs.ratios)
    scene.scribble(4)
    while scene.busy:
        perceiver.observe(scene.tick(), committed)
    i, obs = _stable_within(perceiver, committed, scene)
    assert i is not None
    assert obs.cell_marks[4] == "marked"  # delta vs the committed baseline


def test_two_marks_in_two_cells_both_read_marked():
    scene, perceiver, baseline = _calibrated_scene()
    scene.draw(4, "X")
    while scene.busy:
        perceiver.observe(scene.tick(), baseline)
    scene.draw(0, "X")
    while scene.busy:
        perceiver.observe(scene.tick(), baseline)
    i, obs = _stable_within(perceiver, baseline, scene)
    assert i is not None
    assert obs.cell_marks[0] == "marked" and obs.cell_marks[4] == "marked"


@pytest.mark.parametrize("cell", [6, 8])
def test_a_shadow_on_a_corner_cell_reads_ambiguous_and_only_there(cell):
    # Parametrized over the cells whose geometry calibrates reliably on
    # every background (bottom row, outward diagonal facing the camera
    # side). Cells 0/2's band geometry lands out-of-band on the
    # procedural grid — shadow() raises rather than fake it.
    scene, perceiver, baseline = _calibrated_scene()
    # calibrated against the very Perceiver that will judge it — the
    # ambiguous band is too narrow for cross-context transfer
    scene.shadow(cell, perceiver=perceiver)
    i, obs = _stable_within(perceiver, baseline, scene)
    assert i is not None
    assert obs.cell_marks[cell] == "ambiguous"
    assert all(mark == "none" for j, mark in enumerate(obs.cell_marks) if j != cell)


def test_an_erased_mark_drops_back_below_threshold():
    scene, perceiver, baseline = _calibrated_scene()
    scene.draw(6, "X")
    while scene.busy:
        perceiver.observe(scene.tick(), baseline)
    _, obs = _stable_within(perceiver, baseline, scene)
    committed = list(obs.ratios)
    scene.erase(6)
    while scene.busy:
        perceiver.observe(scene.tick(), committed)
    i, obs = _stable_within(perceiver, committed, scene)
    assert i is not None
    assert obs.ratios[6] - committed[6] < INK_LOW


def test_a_page_bump_loses_the_board_past_the_hold_over_then_recovers():
    scene, perceiver, _baseline = _calibrated_scene(seed=5)
    scene.bump(dx=38, dy=-22, deg=6)
    founds = []
    while scene.busy:
        obs = perceiver.observe(scene.tick(), None)
        founds.append(obs.found)
    # P2's 1.5 s hold-over must expire: found=False at least once means
    # detection was dead long enough to really drive BOARD_LOST.
    lost_run = max((len(list(g)) for k, g in itertools.groupby(founds) if not k), default=0)
    assert lost_run > 0, "detection never dropped past the hold-over — no BOARD_LOST possible"
    i, obs = _stable_within(perceiver, None, scene)
    assert i is not None, "board never recovered after the bump"
    assert obs.found


def test_a_lingering_hand_never_lets_the_scene_go_stable():
    scene, perceiver, baseline = _calibrated_scene(seed=9)
    scene.linger(4.0)
    stables = 0
    while scene.busy:
        obs = perceiver.observe(scene.tick(), baseline)
        stables += bool(obs.stable)
    assert stables == 0, "a hovering hand must count as occluded, never stable"
    # ... and when it leaves, the page is clean again (no residue)
    i, obs = _stable_within(perceiver, baseline, scene)
    assert i is not None
    assert all(mark == "none" for mark in obs.cell_marks)
