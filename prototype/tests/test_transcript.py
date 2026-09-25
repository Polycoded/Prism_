import unittest

from prototype.transcript import normalize_correction, normalize_disfluencies


class TranscriptNormalizationTests(unittest.TestCase):
    def test_normalize_disfluencies_removes_fillers_and_repetitions(self):
        cleaned, changes = normalize_disfluencies("umm what what is uh the LumaPad S1 warranty")
        self.assertEqual(cleaned, "what is the LumaPad S1 warranty")
        self.assertEqual([c["type"] for c in changes], ["filler", "filler", "repetition"])
        self.assertEqual([c["text"] for c in changes], ["umm", "uh", "what"])

    def test_explicit_value_corrections(self):
        examples = [
            ("The room is for 6—sorry, 7 people and what is parking like?", ("The room is for 7 people and what is parking like?", {"type": "explicit_value", "from": "6", "to": "7"}, False)),
            ("6, make that 7", ("7", {"type": "explicit_value", "from": "6", "to": "7"}, False)),
            ("It is 7, not 6 people", ("It is 7 people", {"type": "accepted_value", "from": "6", "to": "7"}, False)),
            ("Book it Tuesday, actually Wednesday and include lunch", ("Book it Wednesday and include lunch", {"type": "explicit_value", "from": "Tuesday", "to": "Wednesday"}, False)),
        ]
        for source, expected in examples:
            with self.subTest(source=source):
                self.assertEqual(normalize_correction(source, {}), expected)

    def test_subject_correction_and_alias_safety(self):
        aliases = {"s1": "LumaPad S1", "s2": "LumaPad S2"}

        self.assertEqual(
            normalize_correction("LumaPad S1 warranty period, sorry I meant S2", aliases),
            ("LumaPad S2 warranty period", {"from": "LumaPad S1", "to": "LumaPad S2"}, False),
        )
        self.assertEqual(
            normalize_correction("LumaPad S1 warranty, sorry I meant S99", aliases),
            ("LumaPad S1 warranty, sorry I meant S99", None, True),
        )
        self.assertEqual(
            normalize_correction("Does LumaPad S1 actually support wireless charging?", aliases),
            ("Does LumaPad S1 actually support wireless charging?", None, False),
        )
        self.assertEqual(
            normalize_correction("Compare LumaPad S1 and LumaPad S2 ... sorry I meant S1", aliases),
            ("Compare LumaPad S1 and LumaPad S2 ... sorry I meant S1", None, True),
        )


if __name__ == "__main__":
    unittest.main()
