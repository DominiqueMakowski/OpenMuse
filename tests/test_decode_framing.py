"""Status framing regressions using synthetic bytes, not private recordings."""

import unittest
from itertools import islice
from pathlib import Path

import numpy as np

from OpenMuse.decode import make_timestamps, parse_message

TIMESTAMP = "2026-01-01T00:00:00+00:00"
UUID = "273e0013-4c4d-454d-96be-f03bac821358"
VARIANTS = (
    (0x11, "EEG", 28, 4, 4),
    (0x12, "EEG", 28, 2, 8),
    (0x34, "OPTICS", 30, 3, 4),
    (0x35, "OPTICS", 40, 2, 8),
    (0x36, "OPTICS", 40, 1, 16),
)


def message(payload):
    return f"{TIMESTAMP}\t{UUID}\t{payload.hex()}"


def packet(tag, body, index=0):
    header = bytearray(14)
    header[0] = len(header) + len(body)
    header[1] = index
    header[9] = tag
    return bytes(header) + body


def subpacket(tag, body):
    return bytes((tag, 0, 0, 0, 0)) + body


def status_body(percent=50):
    return (percent * 256).to_bytes(2, "little") + bytes(18)


class TestStatusFraming(unittest.TestCase):
    def test_status_bytes_cannot_fabricate_mixed_width_eeg(self):
        # Synthetic equivalent of the observed failure: the old parser reads
        # status[0] as EEG8/index 69 and optics[8] as EEG4/index 0. Sorting those
        # fake EEG records then copies a (2, 8) array into a four-column buffer.
        status = bytearray(20)
        status[:2] = bytes((0x12, 69))
        optics = bytearray(40)
        optics[8] = 0x11
        body = bytes(status) + subpacket(0x36, optics) + subpacket(0x12, bytes(28))
        parsed = parse_message(message(packet(0x88, body)))
        self.assertEqual(len(parsed["BATTERY"]), 1)
        self.assertEqual(parsed["BATTERY"][0]["data"][0], 69.0703125)
        self.assertEqual([p["tag_byte"] for p in parsed["EEG"]], [0x12])
        self.assertEqual([p["tag_byte"] for p in parsed["OPTICS"]], [0x36])
        samples, *_ = make_timestamps(parsed["EEG"])
        np.testing.assert_array_equal(samples[:, 1:], np.zeros((2, 8)))

    def test_existing_public_recording_keeps_sensors_after_status(self):
        # Public fixture already in this repository: 14-byte header + 20-byte
        # status + three OPTICS16 subpackets + one ACCGYRO subpacket = 210 bytes.
        path = Path(__file__).parent / "test_data/test_device2.txt"
        with path.open() as source:
            parsed = parse_message(next(islice(source, 855, 856)))
        self.assertEqual(len(parsed["OPTICS"]), 3)
        self.assertEqual(len(parsed["ACCGYRO"]), 1)
        self.assertEqual(parsed["EEG"], [])
        self.assertEqual(len(parsed["BATTERY"]), 1)
        self.assertEqual(parsed["BATTERY"][0]["data"][0], 50.5703125)

    def test_status_can_precede_each_supported_sensor_variant(self):
        for tag, sensor, size, rows, columns in VARIANTS:
            with self.subTest(tag=hex(tag)):
                body = status_body() + subpacket(tag, bytes(size))
                parsed = parse_message(message(packet(0x88, body)))
                self.assertEqual(len(parsed["BATTERY"]), 1)
                self.assertEqual(len(parsed[sensor]), 1)
                sample = parsed[sensor][0]
                self.assertEqual(sample["tag_byte"], tag)
                np.testing.assert_array_equal(sample["data"], np.zeros((rows, columns)))

    def test_embedded_status_preserves_following_sensor_boundaries(self):
        for tag, sensor, size, rows, columns in VARIANTS:
            with self.subTest(tag=hex(tag)):
                body = (bytes(size) + subpacket(0x88, status_body())
                        + subpacket(tag, bytes(size)) + subpacket(0x47, bytes(36)))
                parsed = parse_message(message(packet(tag, body)))
                self.assertEqual([p["tag_byte"] for p in parsed[sensor]], [tag, tag])
                self.assertEqual(len(parsed["BATTERY"]), 1)
                self.assertEqual(len(parsed["ACCGYRO"]), 1)
                for sample in parsed[sensor]:
                    np.testing.assert_array_equal(sample["data"], np.zeros((rows, columns)))

    def test_status_samples_use_one_second_intervals(self):
        payload = packet(0x88, status_body(), 0) + packet(0x88, status_body(), 1)
        parsed = parse_message(message(payload))
        samples, *_ = make_timestamps(parsed["BATTERY"])
        np.testing.assert_allclose(samples, [[0.0, 50.0], [1.0, 50.0]])

    def test_legacy_battery_format_remains_supported(self):
        parsed = parse_message(message(packet(0x98, status_body(42))))
        self.assertEqual(len(parsed["BATTERY"]), 1)
        self.assertEqual(parsed["BATTERY"][0]["tag_byte"], 0x98)
        self.assertEqual(parsed["BATTERY"][0]["data"][0], 42.0)


class TestPacketFraming(unittest.TestCase):
    def assert_empty(self, payload):
        parsed = parse_message(message(payload))
        self.assertTrue(all(not packets for packets in parsed.values()))

    def test_short_declared_lengths_do_not_read_header_fields(self):
        for length in range(14):
            with self.subTest(length=length):
                self.assert_empty(bytes((length,)) + bytes(13))

    def test_truncated_packet_does_not_decode_partial_body(self):
        payload = packet(0x12, bytes(28))
        self.assert_empty(payload[:-1])
        self.assert_empty(payload[:13])

    def test_unknown_primary_does_not_reinterpret_payload_as_tags(self):
        self.assert_empty(packet(0x99, subpacket(0x12, bytes(28))))

    def test_incomplete_primary_does_not_reinterpret_payload_as_tags(self):
        # A complete EEG subpacket is shorter than the required OPTICS16 body.
        self.assert_empty(packet(0x36, subpacket(0x12, bytes(28))))

    def test_short_packet_after_valid_packet_keeps_valid_samples(self):
        parsed = parse_message(message(packet(0x12, bytes(28)) + bytes(14)))
        self.assertEqual(len(parsed["EEG"]), 1)
        np.testing.assert_array_equal(parsed["EEG"][0]["data"], np.zeros((2, 8)))

    def test_truncated_additional_subpacket_keeps_complete_primary(self):
        body = bytes(28) + subpacket(0x36, bytes(40))[:-1]
        parsed = parse_message(message(packet(0x12, body)))
        self.assertEqual(len(parsed["EEG"]), 1)
        self.assertEqual(parsed["OPTICS"], [])
        np.testing.assert_array_equal(parsed["EEG"][0]["data"], np.zeros((2, 8)))
