from cpx_planning.session_sharding import owns_session, session_shard_index


def test_actor_suffix_assigns_sessions_deterministically():
    assert session_shard_index("run-a:25", 2) == 1
    assert session_shard_index("run-a:30", 2) == 0
    assert session_shard_index("run-a:35", 2) == 1
    assert session_shard_index("run-a:40", 2) == 0


def test_run_uuid_does_not_change_actor_ownership():
    assert session_shard_index("first-run:51", 3) == 0
    assert session_shard_index("second-run:51", 3) == 0


def test_non_actor_session_uses_stable_fallback():
    owner = session_shard_index("custom-session", 4)
    assert owner == session_shard_index("custom-session", 4)
    assert owns_session("custom-session", 4, owner)


def test_empty_legacy_session_is_owned_only_by_first_shard():
    assert owns_session("", 2, 0)
    assert not owns_session("", 2, 1)


def test_invalid_shard_index_is_rejected():
    try:
        owns_session("run:1", 2, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid shard index was accepted")
