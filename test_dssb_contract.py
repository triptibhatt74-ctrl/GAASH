"""Local, dependency-free contracts for deliberate DSS-B support."""

import os
import inspect
import asyncio
import unittest
from datetime import date, timedelta
from unittest.mock import AsyncMock, patch


os.environ.setdefault("DATABASE_URL", "postgresql://test.invalid/gaash")
os.environ.setdefault("GAASH_JWT_SECRET", "test-secret-only-not-for-production")

import bot  # noqa: E402


class _CursorResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _Connection:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, *_args):
        return _CursorResult(self._row)


class DSSBContractTests(unittest.TestCase):
    def test_dssb_is_deliberate_scale_with_reference_bounds(self):
        self.assertEqual(bot.normalize_scale("dssb"), "DSS-B")
        self.assertIn("DSS-B", bot.VALID_SCALES)
        self.assertEqual(bot._SCALE_ITEM_COUNT["DSS-B"], 8)
        self.assertEqual(bot._SCORE_MAX["DSS-B"], 4)
        self.assertEqual(bot.compute_total("DSS-B", {i: 4 for i in range(1, 9)}), 32)
        self.assertEqual(bot.interpret_assessment_total("DSS-B", 12), "completed screening score")

    def test_existing_scale_scoring_contract_is_unchanged(self):
        self.assertEqual(bot.compute_total("PHQ-9", {i: 0 for i in range(1, 10)}), 0)
        self.assertEqual(bot.compute_total("GAD-7", {i: 3 for i in range(1, 8)}), 21)
        self.assertEqual(bot.compute_total("PSS-10", {i: 0 for i in range(1, 11)}), 16)

    def test_dssb_is_not_a_conversational_active_scale(self):
        self.assertNotIn("DSS-B", bot.NLPAnalysis.model_fields["active_scale_triggered"].annotation.__args__)
        self.assertEqual(bot.calculate_composite_risk(None, None, None, None, None, False), "UNKNOWN")
        self.assertNotIn("DSS-B", inspect.getsource(bot.chat))

    def test_chat_saves_one_user_turn_and_one_assistant_turn(self):
        source = "".join(inspect.getsource(bot.chat).split())
        self.assertEqual(source.count('save_message(user_id,"user",message_text,conversation_id)'), 1)
        self.assertEqual(source.count('save_message(user_id,"assistant",reply,conversation_id,)'), 1)

    def test_identity_validation_rejects_ordinary_exploration(self):
        signal = bot.IdentityDistressSignal(category="self_concept_confusion", present=True)
        self.assertEqual(bot.validate_identity_distress_signals("I changed my fashion style.", [signal]), [])
        self.assertEqual(bot.validate_identity_distress_signals("I behave differently with my parents and friends.", [signal]), [])

    def test_identity_validation_keeps_explicit_self_disconnection(self):
        signal = bot.IdentityDistressSignal(category="sense_of_self_disconnection", present=True)
        result = bot.validate_identity_distress_signals(
            "I don't know who I am anymore and I feel disconnected from myself.", [signal]
        )
        self.assertEqual(result, [signal])

    def test_adult_gate_requires_stored_date_of_birth(self):
        adult = {"date_of_birth": date.today() - timedelta(days=365 * 20)}
        minor = {"date_of_birth": date.today() - timedelta(days=365 * 17)}
        with patch.object(bot, "get_conn", return_value=_Connection(adult)):
            self.assertEqual(bot._dssb_adult_eligibility_sync(1), (True, ""))
        with patch.object(bot, "get_conn", return_value=_Connection(minor)):
            self.assertEqual(bot._dssb_adult_eligibility_sync(1)[0], False)
        with patch.object(bot, "get_conn", return_value=_Connection(None)):
            self.assertEqual(bot._dssb_adult_eligibility_sync(1)[0], False)

    def test_camera_sequence_contract_is_registered(self):
        paths = {route.path for route in bot.app.routes}
        self.assertIn("/emotion/analyze", paths)
        self.assertIn("/emotion/analyze-sequence", paths)

    def test_single_frame_uses_the_configured_detector(self):
        provider_result = bot.VisualEmotionResult(
            primary="joy",
            confidence=0.8,
            scores={"happy": 80.0},
            status="classified",
        )
        with patch.object(
            bot.visual_emotion_detector,
            "analyze_base64",
            new=AsyncMock(return_value=provider_result),
        ) as analyze:
            result = asyncio.run(bot.analyze_frame("frame"))
        analyze.assert_awaited_once_with("frame")
        self.assertTrue(result.ok)
        self.assertEqual(result.dominant_emotion, "joy")
        self.assertEqual(result.emotion_scores, {"happy": 80.0})

    def test_sequence_averages_successful_single_frame_results(self):
        results = [
            bot.AnalyzeFrameResponse(dominant_emotion="joy", emotion_scores={"joy": 60.0, "sadness": 40.0}, ok=True),
            bot.AnalyzeFrameResponse(dominant_emotion=None, emotion_scores={}, ok=False, error="unavailable"),
            bot.AnalyzeFrameResponse(dominant_emotion="joy", emotion_scores={"joy": 80.0, "sadness": 20.0}, ok=True),
        ]
        with patch.object(bot, "analyze_frame", new=AsyncMock(side_effect=results)):
            result = asyncio.run(bot.analyze_frame_sequence(["a", "b", "c"]))
        self.assertTrue(result.ok)
        self.assertEqual(result.frames_analyzed, 2)
        self.assertEqual(result.dominant_emotion, "joy")
        self.assertEqual(result.emotion_scores, {"joy": 70.0, "sadness": 30.0})


if __name__ == "__main__":
    unittest.main()
