import json

import pytest

from opencda.planning_module.utility import cp_messages


def test_payload_publish_is_atomic_when_serialization_is_interrupted(
    tmp_path, monkeypatch,
):
    message_path = tmp_path / "cp_message.json"
    original = cp_messages.empty_cp_payload(timestamp_s=1.0)
    cp_messages.write_cp_message_payload(original, str(message_path))

    def interrupted_dump(_payload, message_file, indent=None):
        del indent
        message_file.write('{"lane_events": [')
        raise RuntimeError("simulated writer interruption")

    monkeypatch.setattr(cp_messages.json, "dump", interrupted_dump)
    with pytest.raises(RuntimeError, match="writer interruption"):
        cp_messages.write_cp_message_payload(
            {"lane_events": [{"id": "closure"}]}, str(message_path)
        )

    with message_path.open("r", encoding="utf-8") as message_file:
        assert json.load(message_file) == original
    assert list(tmp_path.glob("*.tmp")) == []
