"""
Unit tests for the new decode module.

Tests the parse_message() and decode_rawdata() functions with real BLE message data from Muse devices.
The new decoder returns numpy arrays with timestamps instead of lists of dicts.
"""

import unittest
import os
import numpy as np
from OpenMuse.decode import parse_message, decode_rawdata, SENSORS, make_timestamps


class TestGlobalTimestamping(unittest.TestCase):
    """Test suite for global timestamping logic using decode_rawdata() function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - locate test data files."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")

        # All 15 presets for comprehensive testing
        cls.test_files = {
            "p20": os.path.join(cls.test_data_dir, "data_p20.txt"),
            "p21": os.path.join(cls.test_data_dir, "data_p21.txt"),
            "p50": os.path.join(cls.test_data_dir, "data_p50.txt"),
            "p51": os.path.join(cls.test_data_dir, "data_p51.txt"),
            "p60": os.path.join(cls.test_data_dir, "data_p60.txt"),
            "p61": os.path.join(cls.test_data_dir, "data_p61.txt"),
            "p1034": os.path.join(cls.test_data_dir, "data_p1034.txt"),
            "p1035": os.path.join(cls.test_data_dir, "data_p1035.txt"),
            "p1041": os.path.join(cls.test_data_dir, "data_p1041.txt"),
            "p1042": os.path.join(cls.test_data_dir, "data_p1042.txt"),
            "p1043": os.path.join(cls.test_data_dir, "data_p1043.txt"),
            "p1044": os.path.join(cls.test_data_dir, "data_p1044.txt"),
            "p1045": os.path.join(cls.test_data_dir, "data_p1045.txt"),
            "p1046": os.path.join(cls.test_data_dir, "data_p1046.txt"),
            "p4129": os.path.join(cls.test_data_dir, "data_p4129.txt"),
        }

        # Verify test data files exist
        for preset, filepath in cls.test_files.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Test data file not found: {filepath}\n"
                    f"Please ensure test data files are in {cls.test_data_dir}"
                )

    def test_basic_parsing(self):
        """Test basic parsing and global timestamping of multiple messages."""
        # Read multiple messages from p20 for global timestamping
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            messages = [f.readline().strip() for _ in range(10) if f.readline().strip()]

        result = decode_rawdata(messages)

        # Should return a dict
        self.assertIsInstance(result, dict)

        # Should have all sensor type keys
        expected_keys = {"EEG", "ACCGYRO", "OPTICS", "BATTERY", "Unknown"}
        self.assertEqual(set(result.keys()), expected_keys)

        # Each value should be a DataFrame
        import pandas as pd

        for key, value in result.items():
            self.assertIsInstance(value, pd.DataFrame)

    def test_accgyro_structure(self):
        """Test that ACCGYRO data has correct structure with global timestamping."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            messages = [f.readline().strip() for _ in range(10) if f.readline().strip()]

        result = decode_rawdata(messages)
        accgyro_df = result["ACCGYRO"]

        # Should have data (p20 has ACCGYRO)
        self.assertGreater(len(accgyro_df), 0)

        # Should have 7 columns: time + 6 channels (ACC_X, ACC_Y, ACC_Z, GYRO_X, GYRO_Y, GYRO_Z)
        self.assertEqual(len(accgyro_df.columns), 7)

        # First column should be timestamps (increasing)
        timestamps = np.array(accgyro_df["time"])
        self.assertTrue(
            np.all(np.diff(timestamps) >= 0),
            "Timestamps should be monotonically increasing",
        )

    def test_eeg_structure(self):
        """Test that EEG data has correct structure with global timestamping."""
        with open(self.test_files["p1034"], "r", encoding="utf-8") as f:
            messages = [f.readline().strip() for _ in range(10) if f.readline().strip()]

        result = decode_rawdata(messages)
        eeg_df = result["EEG"]

        # Should have data (p1034 has EEG)
        self.assertGreater(len(eeg_df), 0)

        # Should have time + channels (either 4 or 8)
        self.assertIn(len(eeg_df.columns), [5, 9])  # time + 4 or time + 8

        # First column should be timestamps
        timestamps = np.array(eeg_df["time"])
        self.assertTrue(np.all(np.diff(timestamps) >= 0))

    def test_empty_message(self):
        """Test handling of empty message list."""
        result = decode_rawdata([])

        # Should return empty DataFrames for all sensor types
        import pandas as pd

        for key, value in result.items():
            self.assertIsInstance(value, pd.DataFrame)
            self.assertEqual(len(value), 0)

    def test_timestamp_uniformity(self):
        """Test that timestamps are uniformly spaced at correct sampling rate with global timestamping."""
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            messages = [f.readline().strip() for _ in range(20) if f.readline().strip()]

        result = decode_rawdata(messages)
        accgyro_df = result["ACCGYRO"]

        if len(accgyro_df) > 2:
            timestamps = np.array(accgyro_df["time"])
            deltas = np.diff(timestamps)

            # Expected delta for 52 Hz ACCGYRO
            expected_delta = 1.0 / 52.0

            # With global timestamping, we expect most deltas to be close to expected,
            # but there may be larger gaps between messages
            small_gaps = deltas[
                deltas < expected_delta * 2
            ]  # Gaps smaller than 2x expected

            # At least 80% of gaps should be close to expected delta
            if len(small_gaps) > 0:
                self.assertTrue(
                    np.mean(np.abs(small_gaps - expected_delta) / expected_delta) < 0.1,
                    "Most timestamps should be uniformly spaced",
                )

    def test_sensors_config(self):
        """Test that SENSORS configuration is complete."""
        # Should have all expected sensor TAG bytes
        # 0x88 is a new firmware combined/metadata packet (structure TBD)
        expected_tags = {0x11, 0x12, 0x34, 0x35, 0x36, 0x47, 0x53, 0x88, 0x98}
        self.assertEqual(set(SENSORS.keys()), expected_tags)

        # Each sensor should have required fields
        for tag, config in SENSORS.items():
            self.assertIn("type", config)
            self.assertIn("n_channels", config)
            self.assertIn("n_samples", config)
            self.assertIn("rate", config)
            self.assertIn("data_len", config)

            # Validate types
            self.assertIsInstance(config["type"], str)
            self.assertIsInstance(config["n_channels"], int)
            self.assertIsInstance(config["n_samples"], int)
            self.assertIsInstance(config["rate"], float)
            self.assertIsInstance(config["data_len"], int)

    def test_performance(self):
        """Test that global timestamping decoding is fast enough for real-time use."""
        import time

        # Read 100 messages
        messages = []
        with open(self.test_files["p20"], "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                messages.append(line.strip())

        # Time the decoding with global timestamping
        start = time.perf_counter()
        result = decode_rawdata(messages)
        elapsed = time.perf_counter() - start

        # Should be fast: < 10ms per message on average (global timestamping adds some overhead)
        ms_per_msg = (elapsed / len(messages)) * 1000
        self.assertLess(ms_per_msg, 10.0, f"Decoding too slow: {ms_per_msg:.3f} ms/msg")
        print(f"\nPerformance: {ms_per_msg:.3f} ms/msg")


class TestDecodeRawdata(unittest.TestCase):
    """Test suite for decode_rawdata() function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        cls.test_file = os.path.join(cls.test_data_dir, "data_p20.txt")

    def test_basic_decoding(self):
        """Test basic multi-message decoding to DataFrames."""
        # Read 10 messages
        with open(self.test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)

        # Should return a dict
        self.assertIsInstance(result, dict)

        # Should have all sensor type keys
        expected_keys = {"EEG", "ACCGYRO", "OPTICS", "BATTERY", "Unknown"}
        self.assertEqual(set(result.keys()), expected_keys)

        # Each value should be a DataFrame
        import pandas as pd

        for key, value in result.items():
            self.assertIsInstance(value, pd.DataFrame)

    def test_accgyro_columns(self):
        """Test that ACCGYRO DataFrame has correct columns."""
        with open(self.test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)
        accgyro_df = result["ACCGYRO"]

        # Should have data
        self.assertGreater(len(accgyro_df), 0)

        # Should have correct columns
        expected_columns = [
            "time",
            "ACC_X",
            "ACC_Y",
            "ACC_Z",
            "GYRO_X",
            "GYRO_Y",
            "GYRO_Z",
        ]
        self.assertEqual(list(accgyro_df.columns), expected_columns)

    def test_eeg_columns(self):
        """Test that EEG DataFrame has correct columns."""
        # Use p1034 which has EEG4 data
        test_file = os.path.join(self.test_data_dir, "data_p1034.txt")
        with open(test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)
        eeg_df = result["EEG"]

        # Should have data
        if len(eeg_df) > 0:
            # Should have correct columns for EEG4
            expected_columns = ["time", "EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10"]
            self.assertEqual(list(eeg_df.columns), expected_columns)

    def test_optics_columns(self):
        """Test that Optics DataFrame has correct columns."""
        # Use p1041 which has Optics16 data
        test_file = os.path.join(self.test_data_dir, "data_p1041.txt")
        with open(test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(10)]

        result = decode_rawdata(messages)
        optics_df = result["OPTICS"]

        # Should have data
        if len(optics_df) > 0:
            # Should have correct columns for Optics16
            expected_columns = [
                "time",
                "OPTICS_LO_NIR",
                "OPTICS_RO_NIR",
                "OPTICS_LO_IR",
                "OPTICS_RO_IR",
                "OPTICS_LI_NIR",
                "OPTICS_RI_NIR",
                "OPTICS_LI_IR",
                "OPTICS_RI_IR",
                "OPTICS_LO_RED",
                "OPTICS_RO_RED",
                "OPTICS_LO_AMB",
                "OPTICS_RO_AMB",
                "OPTICS_LI_RED",
                "OPTICS_RI_RED",
                "OPTICS_LI_AMB",
                "OPTICS_RI_AMB",
            ]
            self.assertEqual(list(optics_df.columns), expected_columns)

    def test_concatenation(self):
        """Test that multiple messages are properly concatenated with global timestamping."""
        # Read messages
        with open(self.test_file, "r", encoding="utf-8") as f:
            messages = [f.readline() for _ in range(5)]

        # Count total samples across all subpackets from all messages
        total_samples = 0
        for message in messages:
            parsed = parse_message(message)
            for subpacket in parsed["ACCGYRO"]:
                total_samples += subpacket["n_samples"]

        # Parse all messages together with global timestamping
        result = decode_rawdata(messages)
        combined_total = len(result["ACCGYRO"])

        # Should have same total samples
        self.assertEqual(total_samples, combined_total)


class TestBattery(unittest.TestCase):
    """Test suite for battery decoding."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        cls.battery_files = {
            "16_80": os.path.join(cls.test_data_dir, "test_battery_16_80.txt"),
            "58_27": os.path.join(cls.test_data_dir, "test_battery_58_27.txt"),
            "90_40": os.path.join(cls.test_data_dir, "test_battery_90_40.txt"),
        }

    def test_battery_known_levels(self):
        """Test battery decoding with files at known battery levels using global timestamping."""
        expected_values = {
            "16_80": 16.80,
            "58_27": 58.27,
            "90_40": 90.40,
        }

        for key, filepath in self.battery_files.items():
            # Read all messages and decode with global timestamping
            with open(filepath, "r", encoding="utf-8") as f:
                messages = [line.strip() for line in f if line.strip()]

            result = decode_rawdata(messages)
            battery_df = result["BATTERY"]

            # Should have found at least one battery sample
            self.assertGreater(len(battery_df), 0, f"No battery data found in {key}")

            # Check first battery value
            battery_percent = battery_df.iloc[
                0, 1
            ]  # First row, second column (after time)
            expected = expected_values[key]

            # Should be within 0.5% of expected
            self.assertAlmostEqual(
                float(battery_percent),
                expected,
                delta=0.5,
                msg=f"Battery level mismatch for {key}: got {float(battery_percent):.2f}, expected {expected:.2f}",
            )


if __name__ == "__main__":
    unittest.main()
