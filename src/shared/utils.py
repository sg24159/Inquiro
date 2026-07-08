import re

_LIST_MARKER = re.compile(r'^[\-\*\+]\s+')
_NUMBERED_MARKER = re.compile(r'^\d+[\.\)]\s*')


def strip_line_noise(line: str) -> str:
    line = line.strip()
    line = _LIST_MARKER.sub('', line)
    line = _NUMBERED_MARKER.sub('', line)
    return line
