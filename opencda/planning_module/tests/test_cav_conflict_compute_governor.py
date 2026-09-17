from pipeline.cav_conflict_compute_governor import CAVConflictComputeGovernor


def _governor(**overrides):
    config = dict(
        max_agents=6, max_modes=3, min_agents=2, min_modes=1,
        budget_ms=100.0, window=5, degrade_streak=3, recover_streak=5,
        recover_headroom_ratio=0.5,
    )
    config.update(overrides)
    return CAVConflictComputeGovernor(**config)


def test_starts_at_the_static_ceiling():
    gov = _governor()
    assert gov.current_max_relevant_agents == 6
    assert gov.current_max_modes_per_agent == 3


def test_holds_with_insufficient_history():
    gov = _governor()
    for _ in range(4):
        decision = gov.observe_stage_ms(500.0)
    assert decision.reason == "insufficient_history"
    assert not decision.degraded


def test_sustained_overrun_degrades_modes_before_agents():
    gov = _governor()
    # window=5 fills on the 5th call, which is also the first over-budget
    # streak tick (streak=1); degrade_streak=3 means streak reaches 3 -- and
    # trips -- on the 7th call, so the 6th is the last one still holding.
    for _ in range(6):
        decision = gov.observe_stage_ms(150.0)
    assert gov.current_max_modes_per_agent == 3
    assert not decision.degraded
    decision = gov.observe_stage_ms(150.0)
    assert decision.degraded
    assert gov.current_max_modes_per_agent == 2
    assert gov.current_max_relevant_agents == 6


def test_one_off_spike_amid_fast_ticks_does_not_degrade():
    gov = _governor()
    # A single slow tick, far enough from any other slow tick that no
    # trailing 5-tick window contains it more than once -- unlike a spike
    # recurring exactly every `window` ticks, which would keep every window
    # average over budget forever and isn't a "one-off" at all.
    pattern = [10.0] * 8 + [500.0] + [10.0] * 20
    for stage_ms in pattern:
        decision = gov.observe_stage_ms(stage_ms)
    assert not decision.degraded
    assert gov.current_max_relevant_agents == 6
    assert gov.current_max_modes_per_agent == 3


def test_recovers_agents_before_modes_once_headroom_sustained():
    gov = _governor()
    # Degrade far enough that both modes (floor 1) and agents (below the
    # ceiling of 6) have stepped down -- step-down prioritizes modes, so
    # modes must already be at its floor before an agent is ever given up.
    while gov.current_max_relevant_agents == 6:
        gov.observe_stage_ms(150.0)
    assert gov.current_max_modes_per_agent == 1
    degraded_agents = gov.current_max_relevant_agents
    assert degraded_agents < 6

    # Recovery must restore the agent count before touching modes again.
    while gov.current_max_relevant_agents < 6:
        gov.observe_stage_ms(10.0)
    assert gov.current_max_relevant_agents == 6
    assert gov.current_max_modes_per_agent == 1

    # Each step-up (agent or mode) resets the recovery streak, so the last
    # agent step-up leaves modes needing its own full recover_streak of
    # consecutive headroom before it moves.
    for _ in range(4):
        gov.observe_stage_ms(10.0)
    assert gov.current_max_modes_per_agent == 1
    gov.observe_stage_ms(10.0)
    assert gov.current_max_modes_per_agent == 2


def test_never_degrades_below_the_configured_floor():
    gov = _governor(max_agents=2, max_modes=1, min_agents=2, min_modes=1)
    for _ in range(30):
        decision = gov.observe_stage_ms(1000.0)
    assert gov.current_max_relevant_agents == 2
    assert gov.current_max_modes_per_agent == 1
    assert not decision.degraded


def test_reset_returns_to_the_static_ceiling():
    gov = _governor()
    for _ in range(30):
        gov.observe_stage_ms(150.0)
    assert gov.current_max_modes_per_agent < 3
    gov.reset()
    assert gov.current_max_relevant_agents == 6
    assert gov.current_max_modes_per_agent == 3


def test_between_headroom_and_budget_holds_without_flapping():
    gov = _governor()
    # 75ms is above the 50ms recovery floor but below the 100ms budget.
    for _ in range(10):
        decision = gov.observe_stage_ms(75.0)
    assert decision.reason == "holding"
    assert gov.current_max_relevant_agents == 6
    assert gov.current_max_modes_per_agent == 3
