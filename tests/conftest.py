import pytest

from sciagent.spec import StudySpec


@pytest.fixture
def make_spec():
    def _make(sim="reactor", kind="sh", budget=120, **kw):
        planner = kw.pop("planner", {"kind": kind})
        return StudySpec(name=f"t-{sim}-{kind}", simulator=sim, budget={"compute_units": budget}, batch_size=4,
                         planner=planner, **kw)
    return _make
