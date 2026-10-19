import contextlib
import io
import unittest

import numpy as np

from OpenMuse.view_rerun import _gravity_rotation


class GravityRotationTests(unittest.TestCase):
    def test_level_has_no_rotation(self):
        rotation = _gravity_rotation(np.array([0., 0., 1.]))
        assert rotation is not None
        np.testing.assert_allclose(rotation, np.eye(3))

    def test_roll_and_pitch_signs(self):
        cases = [
            ([0., 1., 0.], [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
            ([1., 0., 0.], [[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]),
        ]
        for acceleration, expected in cases:
            with self.subTest(acceleration=acceleration):
                rotation = _gravity_rotation(np.array(acceleration))
                assert rotation is not None
                np.testing.assert_allclose(rotation, expected, atol=1e-12)

    def test_rotation_aligns_gravity_without_scaling_or_reflection(self):
        for acceleration in ([1., 2., 3.], [-2., 1., -3.], [0., 0., -1.]):
            with self.subTest(acceleration=acceleration):
                vector = np.array(acceleration)
                rotation = _gravity_rotation(vector)
                assert rotation is not None
                np.testing.assert_allclose(rotation @ vector,
                                           [0., 0., float(np.linalg.norm(vector))], atol=1e-12)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(rotation), 1.)
                scaled = _gravity_rotation(vector * 100)
                assert scaled is not None
                np.testing.assert_allclose(rotation, scaled, atol=1e-12)

    def test_invalid_acceleration_has_no_orientation(self):
        for acceleration in ([0., 0., 0.], [1e-10, 0., 0.],
                             [np.nan, 0., 1.], [0., np.inf, 1.]):
            with self.subTest(acceleration=acceleration):
                self.assertIsNone(_gravity_rotation(np.array(acceleration)))


class ViewerCliValidationTests(unittest.TestCase):
    def test_headless_viewing_requires_a_recording_destination(self):
        from OpenMuse.cli import main

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["view-rerun", "--no-spawn"])
        self.assertEqual(error.exception.code, 2)

    def test_nonpositive_or_nonfinite_duration_is_rejected(self):
        from OpenMuse.cli import main

        for duration in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(duration=duration):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        main(["view-rerun", f"--duration={duration}"])
                self.assertEqual(error.exception.code, 2)
