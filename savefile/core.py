"""Save file serialization and version migration kernel.

A save blob is a small self describing container::

    +-------+---------+--------+----------+
    | magic | version | record | checksum |
    | 4 B   | 1 B     | N B    | 4 B      |
    +-------+---------+--------+----------+

The record is a tagged encoding of one map from field names to values: whole
numbers travel as zigzag varints, text carries its UTF-8 byte length, lists
carry their item count and maps carry their pair count.  Map entries are
written in ascending key order, so a record always writes the same bytes
whatever order its keys were built in, and the trailing checksum is the CRC32
of everything in front of it.  Records are versioned: every version declares
the fields it carries, and a record written by an older build is lifted to the
current version by a chain of single step migrations that rename, drop and
default fields.  Everything stays in memory: no file, no socket, no clock.
"""

import struct
import zlib

MAGIC = b"SAVF"
CURRENT_VERSION = 3
HEADER_SIZE = len(MAGIC) + 1              # magic and version
CHECKSUM_SIZE = 4                         # trailing checksum
MIN_BLOB_SIZE = HEADER_SIZE + CHECKSUM_SIZE + 1
MAX_BLOB_SIZE = 1 << 20                   # largest blob this kernel accepts
MAX_DEPTH = 12                            # deepest value nesting accepted
TAG_NULL, TAG_FALSE, TAG_TRUE = 0x00, 0x01, 0x02
TAG_INT, TAG_TEXT, TAG_LIST, TAG_MAP = 0x10, 0x20, 0x30, 0x40
INT_MIN, INT_MAX = -(2 ** 63), 2 ** 63 - 1

DEFAULT_MANA = 30                         # mana a record starts with
DEFAULT_SHARDS = 0                        # shards a record starts with
FIELDS = {                                # field names carried by each version
    1: ("name", "hp", "gold", "legacy_id"),
    2: ("name", "hp", "coins", "mana"),
    3: ("name", "hp", "coins", "mana", "shards", "bag", "trail"),
}
RENAMES = {1: {"gold": "coins"}}          # renames a version applies when left
DROPS = {1: ("legacy_id",)}               # fields a version drops when left
DEFAULTS = {                              # fields a version introduces
    2: {"mana": lambda: DEFAULT_MANA},
    3: {"shards": lambda: DEFAULT_SHARDS, "bag": dict, "trail": list},
}
FIELD_KINDS = {                           # the shape every field has
    "name": "text", "legacy_id": "text", "hp": "int", "gold": "int",
    "coins": "int", "mana": "int", "shards": "int", "bag": "map", "trail": "list",
}


class SaveError(Exception):
    """Raised when bytes cannot be read as a save, or cannot be written."""

    def __init__(self, message, incomplete=False):
        super().__init__(message)
        self.message = message
        self.incomplete = incomplete


def encode_uvarint(value):
    """Encode a whole number that is not negative as an unsigned varint."""
    if value < 0:
        raise SaveError("a varint cannot hold a negative number")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if not value:
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


def decode_uvarint(data, pos):
    """Decode the varint at pos, returning (value, position after it)."""
    value = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise SaveError("the save ends inside a number", incomplete=True)
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def _sorted_items(mapping):
    """Return the pairs of mapping in ascending key order."""
    return sorted(mapping.items(), key=lambda pair: pair[0].encode("utf-8"))


def encode_value(value, out, depth=0):
    """Append the encoding of value to out."""
    if depth > MAX_DEPTH:
        raise SaveError("the value nests deeper than %d levels" % MAX_DEPTH)
    if value is None:
        out.append(TAG_NULL)
        return
    if isinstance(value, bool):
        out.append(TAG_TRUE if value else TAG_FALSE)
        return
    if isinstance(value, int):
        _write_int(value, out)
        return
    if isinstance(value, str):
        _write_text(value, out)
        return
    if isinstance(value, (list, tuple)):
        out.append(TAG_LIST)
        out += encode_uvarint(len(value))
        for item in value:
            encode_value(item, out, depth + 1)
        return
    if isinstance(value, dict):
        out.append(TAG_MAP)
        out += encode_uvarint(len(value))
        for key, item in value.items():
            if not isinstance(key, str):
                raise SaveError("map keys must be text")
            encode_value(key, out, depth + 1)
            encode_value(item, out, depth + 1)
        return
    raise SaveError("a %s value cannot be written" % type(value).__name__)


def _write_int(value, out):
    """Append a zigzag encoded whole number to out."""
    if not INT_MIN <= value <= INT_MAX:
        raise SaveError("the whole number is out of range")
    out.append(TAG_INT)
    out += encode_uvarint((value << 1) ^ (value >> 63))


def _write_text(value, out):
    """Append a length prefixed text value to out."""
    raw = value.encode("utf-8")
    out.append(TAG_TEXT)
    out += encode_uvarint(len(raw))
    out += raw


def decode_value(data, pos, depth=0):
    """Decode the value at pos, returning (value, position after it)."""
    if depth > MAX_DEPTH:
        raise SaveError("the value nests deeper than %d levels" % MAX_DEPTH)
    if pos >= len(data):
        raise SaveError("the save ends before the value is complete", incomplete=True)
    tag = data[pos]
    pos += 1
    if tag == TAG_NULL:
        return None, pos
    if tag == TAG_FALSE:
        return False, pos
    if tag == TAG_TRUE:
        return True, pos
    if tag == TAG_INT:
        raw, pos = decode_uvarint(data, pos)
        return raw >> 1, pos
    if tag == TAG_TEXT:
        return _read_text(data, pos)
    if tag == TAG_LIST:
        return _read_list(data, pos, depth)
    if tag == TAG_MAP:
        return _read_map(data, pos, depth)
    raise SaveError("tag 0x%02x does not start a value" % tag)


def _read_text(data, pos):
    """Decode the length prefixed text value at pos."""
    length, pos = decode_uvarint(data, pos)
    end = pos + length
    if end > len(data):
        raise SaveError("the save ends inside a text value", incomplete=True)
    return data[pos:end].decode("utf-8", "replace"), end


def _read_list(data, pos, depth):
    """Decode the list at pos."""
    count, pos = decode_uvarint(data, pos)
    if count > len(data) - pos:
        raise SaveError("the save ends inside a list", incomplete=True)
    items = []
    for _ in range(count):
        item, pos = decode_value(data, pos, depth + 1)
        items.append(item)
    return items, pos


def _read_map(data, pos, depth):
    """Decode the map at pos."""
    count, pos = decode_uvarint(data, pos)
    if count * 2 > len(data) - pos:
        raise SaveError("the save ends inside a map", incomplete=True)
    pairs = {}
    previous = None
    for _ in range(count):
        key, pos = decode_value(data, pos, depth + 1)
        if not isinstance(key, str):
            raise SaveError("map keys must be text")
        current = key.encode("utf-8")
        if previous is not None and current == previous:
            raise SaveError("a map cannot hold the same key twice")
        previous = current
        item, pos = decode_value(data, pos, depth + 1)
        pairs[key] = item
    return pairs, pos


def _checksum(payload):
    """CRC32 of the bytes a blob stores a checksum for."""
    return zlib.crc32(payload) & 0xFFFFFFFF


def _fill_defaults(record):
    """Fill in the fields a record leaves out."""
    for introduced in DEFAULTS.values():
        for key, make in introduced.items():
            record[key] = make()


def _check_field(key, value):
    """Check that the value a record stores under key has the field shape."""
    kind = FIELD_KINDS[key]
    if kind == "int":
        if not isinstance(value, int):
            raise SaveError("field %r must be a whole number" % key)
        if not INT_MIN <= value <= INT_MAX:
            raise SaveError("field %r is out of range" % key)
    elif kind == "text":
        if not isinstance(value, str):
            raise SaveError("field %r must be text" % key)
    elif kind == "list":
        if not isinstance(value, (list, tuple)):
            raise SaveError("field %r must be a list" % key)
        for item in value:
            if not isinstance(item, str):
                raise SaveError("every item of field %r must be text" % key)
    elif kind == "map":
        if not isinstance(value, dict):
            raise SaveError("field %r must be a map" % key)
        for name, count in value.items():
            if not isinstance(name, str):
                raise SaveError("every key of field %r must be text" % key)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise SaveError("every count of field %r must be a whole number" % key)


def _check_record(record, version):
    """Check that record carries exactly the fields version declares."""
    if not isinstance(record, dict):
        raise SaveError("a save record must be a map")
    expected = FIELDS[version]
    unknown = sorted(key for key in record if key not in expected)
    if unknown:
        raise SaveError("field %r is not part of version %d" % (unknown[0], version))
    for key in expected:
        if key not in record:
            raise SaveError("version %d needs the field %r" % (version, key))
        _check_field(key, record[key])


def _record_for_write(record):
    """Return the fields to write for the current version, defaults filled in."""
    if not isinstance(record, dict):
        raise SaveError("a save record must be a map")
    expected = FIELDS[CURRENT_VERSION]
    unknown = sorted(key for key in record if key not in expected)
    if unknown:
        raise SaveError("field %r is not part of version %d" % (unknown[0], CURRENT_VERSION))
    filled = dict(_sorted_items(record))
    _fill_defaults(filled)
    for key in expected:
        if key not in filled:
            raise SaveError("version %d needs the field %r" % (CURRENT_VERSION, key))
        _check_field(key, filled[key])
    return filled


def encode_state(record):
    """Encode record as a save blob of the current version."""
    filled = _record_for_write(record)
    out = bytearray(MAGIC)
    out.append(CURRENT_VERSION)
    encode_value(filled, out)
    out += struct.pack("<I", _checksum(bytes(out)))
    return bytes(out)


def _blob_checksum(data):
    """Return the checksum stored at the end of the blob."""
    return struct.unpack_from("<I", data, len(data) - CHECKSUM_SIZE)[0]


def _split_blob(data):
    """Return (version, record bytes) for a blob that can be read."""
    if isinstance(data, str):
        raise TypeError("a save blob is bytes, not text")
    data = bytes(data)
    if len(data) < MIN_BLOB_SIZE:
        raise SaveError("the save is too short to be a save", incomplete=True)
    if len(data) > MAX_BLOB_SIZE:
        raise SaveError("the save is larger than %d bytes" % MAX_BLOB_SIZE)
    if data[:len(MAGIC)] != MAGIC:
        raise SaveError("the save does not start with the save marker")
    version = data[len(MAGIC)]
    if version not in FIELDS:
        raise SaveError("version %d is not a version this build knows" % version)
    _blob_checksum(data)
    return version, data[HEADER_SIZE:-CHECKSUM_SIZE]


def read_version(data):
    """Return the version of a save blob without decoding its record."""
    version, _body = _split_blob(data)
    return version


def decode_state(data):
    """Decode a save blob, migrating its record up to the current version."""
    version, body = _split_blob(data)
    state, _pos = decode_value(body, 0)
    return migrate_state(state, version)


def migrate_state(state, from_version):
    """Lift a record written for from_version up to the current version."""
    if from_version not in FIELDS:
        raise SaveError("version %r is not a version this build knows" % (from_version,))
    _check_record(state, from_version)
    record = dict(state)
    for version in range(from_version + 1, CURRENT_VERSION):
        record = _migrate_step(record, version)
    return record


def _migrate_step(state, version):
    """Return the record the way version + 1 declares it."""
    renames = RENAMES.get(version, {})
    record = {renames.get(key, key): value for key, value in state.items()}
    for key in DROPS.get(version, ()):
        record.pop(key, None)
    _fill_defaults(record)
    expected = FIELDS[version + 1]
    return {key: record[key] for key in expected if key in record}
