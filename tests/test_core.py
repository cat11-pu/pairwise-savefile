"""Behaviour tests for the save file serializer and migration kernel.

The tests only describe what a caller of the kernel is allowed to observe:
the bytes a save writes, the record that comes back, and the errors reported
for saves that are damaged or written for a version this build knows.  Run
them from the project root:

    python3 -m unittest discover -s tests -v
"""

import os
import struct
import sys
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import savefile.core as save


#: Every field of the current version, in ascending order.
FIELDS_NOW = ["bag", "coins", "hp", "mana", "name", "shards", "trail"]


class Raw:
    """A value a test writes as raw bytes, bypassing the encoder."""

    def __init__(self, data):
        self.data = data


def _uvarint(value):
    """Encode a whole number that is not negative the way saves store it."""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode(value, sort=True):
    """Encode a value the way saves store it."""
    if isinstance(value, Raw):
        return value.data
    if value is None:
        return b"\x00"
    if value is True:
        return b"\x01"
    if value is False:
        return b"\x02"
    if isinstance(value, int):
        return b"\x10" + _uvarint((value << 1) ^ (value >> 63))
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"\x20" + _uvarint(len(raw)) + raw
    if isinstance(value, (list, tuple)):
        return b"\x30" + _uvarint(len(value)) + b"".join(
            _encode(item, sort) for item in value
        )
    if isinstance(value, dict):
        pairs = list(value.items())
        if sort:
            pairs.sort(key=lambda pair: pair[0].encode("utf-8"))
        return b"\x40" + _uvarint(len(pairs)) + b"".join(
            _encode(key, sort) + _encode(item, sort) for key, item in pairs
        )
    raise AssertionError("a test cannot build a %s value" % type(value).__name__)


def blob(version, record, junk=b"", sort=True):
    """Assemble a save blob straight from the wire format, checksum included."""
    head = save.MAGIC + bytes([version]) + _encode(record, sort) + junk
    return head + struct.pack("<I", zlib.crc32(head) & 0xFFFFFFFF)


def record(**overrides):
    """A complete record of the current version."""
    filled = {
        "name": "aria",
        "hp": 42,
        "coins": 250,
        "mana": 99,
        "shards": 3,
        "bag": {"rope": 1, "potion": 3},
        "trail": ["village", "cave"],
    }
    filled.update(overrides)
    return filled


class RoundTripTests(unittest.TestCase):
    """Saves written and read back by the same build."""

    def test_a_current_version_save_round_trips_byte_for_byte(self):
        written = record()
        raw = save.encode_state(written)
        self.assertEqual(save.read_version(raw), save.CURRENT_VERSION)
        again = save.decode_state(raw)
        self.assertEqual(sorted(again), sorted(written))
        self.assertEqual(again.get("name"), "aria")
        self.assertEqual(again.get("mana"), 99)
        self.assertEqual(again.get("bag"), {"rope": 1, "potion": 3})
        self.assertEqual(again.get("trail"), ["village", "cave"])
        self.assertEqual(save.encode_state(again), raw)

    def test_a_save_that_leaves_the_later_fields_out_gets_the_defaults(self):
        raw = save.encode_state({"name": "aria", "hp": 42, "coins": 250})
        again = save.decode_state(raw)
        self.assertEqual(sorted(again), sorted(FIELDS_NOW))
        self.assertEqual(again.get("coins"), 250)
        self.assertEqual(again.get("mana"), save.DEFAULT_MANA)
        self.assertEqual(again.get("shards"), save.DEFAULT_SHARDS)
        self.assertEqual(again.get("bag"), {})
        self.assertEqual(again.get("trail"), [])

    def test_reordering_the_keys_of_a_map_does_not_change_the_bytes(self):
        first = record(bag={"rope": 1, "potion": 3})
        second = record(bag={"potion": 3, "rope": 1})
        self.assertEqual(save.encode_state(first), save.encode_state(second))
        again = save.decode_state(save.encode_state(second))
        self.assertEqual(again.get("bag"), {"potion": 3, "rope": 1})

    def test_negative_numbers_survive_the_round_trip(self):
        raw = save.encode_state(record(hp=-7))
        again = save.decode_state(raw)
        self.assertEqual(again.get("hp"), -7)
        self.assertEqual(sorted(again), sorted(FIELDS_NOW))


class ChecksumTests(unittest.TestCase):
    """The checksum has to cover the bytes stored in front of it."""

    def test_editing_one_byte_of_a_record_is_reported(self):
        raw = bytearray(save.encode_state(record()))
        self.assertEqual(
            struct.unpack("<I", bytes(raw[-4:]))[0],
            zlib.crc32(bytes(raw[:-4])) & 0xFFFFFFFF,
        )
        at = raw.index(b"aria")
        raw[at] ^= 0x20
        self.assertRaises(save.SaveError, save.decode_state, bytes(raw))


class MigrationTests(unittest.TestCase):
    """Records written by older builds."""

    def test_a_version_one_save_is_migrated_through_the_whole_chain(self):
        raw = blob(1, {"name": "aria", "hp": 12, "gold": 250, "legacy_id": "old-1"})
        again = save.decode_state(raw)
        self.assertEqual(sorted(again), sorted(FIELDS_NOW))
        self.assertEqual(again.get("name"), "aria")
        self.assertEqual(again.get("hp"), 12)
        self.assertEqual(again.get("coins"), 250)
        self.assertNotIn("gold", again)
        self.assertNotIn("legacy_id", again)
        self.assertEqual(again.get("mana"), save.DEFAULT_MANA)
        self.assertEqual(again.get("shards"), save.DEFAULT_SHARDS)
        self.assertEqual(again.get("bag"), {})
        self.assertEqual(again.get("trail"), [])

    def test_a_version_two_save_keeps_the_values_it_carries(self):
        raw = blob(2, {"name": "aria", "hp": 12, "coins": 250, "mana": 40})
        again = save.decode_state(raw)
        self.assertEqual(again.get("name"), "aria")
        self.assertEqual(again.get("hp"), 12)
        self.assertEqual(again.get("coins"), 250)
        self.assertEqual(again.get("mana"), 40)


class DamagedSaveTests(unittest.TestCase):
    """Bytes that must be reported instead of being accepted or dropped."""

    def test_a_map_whose_keys_are_out_of_order_is_rejected(self):
        descending = {
            "trail": ["village"],
            "shards": 3,
            "name": "aria",
            "mana": 99,
            "hp": 42,
            "coins": 250,
            "bag": {"rope": 1, "potion": 3},
        }
        self.assertRaises(save.SaveError, save.decode_state, blob(3, descending, sort=False))
        self.assertEqual(save.decode_state(blob(3, descending)).get("coins"), 250)

    def test_a_body_the_format_forbids_is_rejected(self):
        with_junk = blob(3, record(), junk=b"\x00")
        self.assertRaises(save.SaveError, save.decode_state, with_junk)
        broken_text = blob(3, record(name=Raw(b"\x20" + _uvarint(4) + b"ari\x80")))
        self.assertRaises(save.SaveError, save.decode_state, broken_text)
        unknown_field = blob(3, record(weapon="sword"))
        self.assertRaises(save.SaveError, save.decode_state, unknown_field)

    def test_a_flag_where_an_integer_belongs_is_rejected(self):
        self.assertRaises(save.SaveError, save.decode_state, blob(3, record(hp=True)))
        self.assertEqual(save.decode_state(blob(3, record(hp=1))).get("hp"), 1)


if __name__ == "__main__":
    unittest.main()
