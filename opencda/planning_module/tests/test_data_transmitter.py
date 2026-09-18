"""opencda.data_transmitter.send() -- the OpenCDA-side half of the CP-ROS-WS
TCP data bridge (github.com/showmen78/CP-ROS-WS). Its cooperative_message_
publisher.py already parses traffic_controls/control into a typed
TrafficControl ROS message, but send() only ever forwarded lane_events --
this pins that it now also forwards the CP payload's "control" list.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from opencda import data_transmitter


class _FakeConnection:
    def __init__(self):
        self.sent = []

    def settimeout(self, _value):
        pass

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        pass


def _decode_stream(connection, stream_name):
    for raw in connection.sent:
        message = json.loads(raw.decode("utf-8"))
        if message.get("message_type") == stream_name:
            return message["data"]
    return None


class DataTransmitterCooperativeStreamTests(unittest.TestCase):
    def setUp(self):
        data_transmitter.close()

    def tearDown(self):
        data_transmitter.close()

    def test_send_forwards_traffic_controls_alongside_lane_events(self):
        connection = _FakeConnection()
        localization_transform = SimpleNamespace(
            location=SimpleNamespace(x=1.0, y=2.0, z=0.3),
            rotation=SimpleNamespace(yaw=0.0),
        )
        cp_payload = {
            "schema_version": 1,
            "timestamp_s": 5.0,
            "lane_events": [{
                "id": "closure-1", "type": "lane_closure",
                "position": [10.0, 4.0, 0.0], "confidence": 1.0,
            }],
            "control": [{
                "id": "signal-1", "source": "roadside_unit",
                "signal_state": "red", "confidence": 0.9,
            }],
        }

        with patch.object(
            data_transmitter.socket, "create_connection",
            return_value=connection,
        ):
            results = data_transmitter.send(
                localization_transform=localization_transform,
                localization_speed_kmh=0.0,
                perception_objects={"vehicles": [], "traffic_lights": []},
                cp_payload=cp_payload,
                cooperative_payload=cp_payload,
                timestamp_s=5.0,
            )

        self.assertTrue(results["cooperative"])
        cooperative_data = _decode_stream(connection, "cooperative")
        self.assertEqual(len(cooperative_data["lane_events"]), 1)
        self.assertEqual(cooperative_data["lane_events"][0]["id"], "closure-1")
        self.assertEqual(len(cooperative_data["traffic_controls"]), 1)
        self.assertEqual(
            cooperative_data["traffic_controls"][0]["id"], "signal-1"
        )
        self.assertEqual(
            cooperative_data["traffic_controls"][0]["signal_state"], "red"
        )

    def test_send_defaults_traffic_controls_to_empty_list(self):
        connection = _FakeConnection()
        localization_transform = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        )

        with patch.object(
            data_transmitter.socket, "create_connection",
            return_value=connection,
        ):
            data_transmitter.send(
                localization_transform=localization_transform,
                localization_speed_kmh=0.0,
                perception_objects={"vehicles": [], "traffic_lights": []},
                cp_payload=None,
                cooperative_payload=None,
                timestamp_s=1.0,
            )

        cooperative_data = _decode_stream(connection, "cooperative")
        self.assertEqual(cooperative_data["traffic_controls"], [])


if __name__ == "__main__":
    unittest.main()
