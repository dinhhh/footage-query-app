# from app import extract_timestamps
import re

# Same range rules as app.TIMESTAMP_PATTERN (whole-text scan; **…** around times still match).
_RANGE_SEP = r"\s*[-\u2013\u2014]\s*"
_EXTRACT_TIMESTAMPS_PATTERN = re.compile(
    rf"(\d{{1,2}}:\d{{2}}(?::\d{{2}})?)(?:{_RANGE_SEP}(\d{{1,2}}:\d{{2}}(?::\d{{2}})?))?"
)


def _ts_part_to_seconds(part: str) -> int:
    nums = [int(p) for p in part.split(":")]
    if len(nums) == 3:
        h, m, s = nums
        return h * 3600 + m * 60 + s
    if len(nums) == 2:
        m, s = nums
        return m * 60 + s
    raise ValueError(f"Unexpected timestamp format: {part}")


def extract_timestamps(text: str) -> list[tuple[int, int]]:
    """
    Extracts all timestamps from the given text and returns them as a list of (start, end) tuples in seconds.
    Supported forms include:
    - Single: ``0:01``, ``00:00``, ``00:00:03``
    - Tight ranges: ``0:01-0:02``, ``Clip 1 (0:00-0:15)``
    - Spaced ranges: ``00:01 - 00:02``, ``00:00:03 - 00:00:06``
    - Markdown bullets: ``**0:01 - 0:02:**`` (extra ``:`` after the range is ignored)
    Range separator may be ASCII ``-``, en dash, or em dash. Results follow document order.
    """
    out: list[tuple[int, int]] = []
    for m in _EXTRACT_TIMESTAMPS_PATTERN.finditer(text):
        a = _ts_part_to_seconds(m.group(1))
        if m.group(2):
            b = _ts_part_to_seconds(m.group(2))
            out.append((a, b))
        else:
            out.append((a, a))
    return out

if __name__ == "__main__":
    # Option 1
    test_text_1 = """
Option 1: In this video, there are 8 turning vehicles.

**Clip 1 (0:00-0:15):**
- **0:01-0:02**: A white car turns right.
- **0:02-0:03**: A grey SUV turns right.
- **0:04-0:05**: A white truck with green crates turns left.
- **0:07-0:08**: A dark blue car turns left.
- **0:09-0:10**: A white car turns right.
- **0:10-0:11**: A silver car turns right.
- **0:13-0:14**: A white pickup truck turns right.
- **0:14-0:15**: A white car turns left.

Return = 8 timestamps for each turning vehicles.
"""

    assert extract_timestamps(test_text_1) == [
        (0, 15),
        (1, 2),
        (2, 3),
        (4, 5),
        (7, 8),
        (9, 10),
        (10, 11),
        (13, 14),
        (14, 15),
    ], extract_timestamps(test_text_1)

    # Option 2
    test_text_2 = """
Option 2: There are **3** white cars in the video.

Here are the details:
1.  **00:00**: A white car (possibly a security vehicle with blue/purple markings) is seen turning left in the intersection.
2.  **00:05**: A white flatbed truck with green cages on the back enters from the left side of the frame and proceeds straight.
3.  **00:13**: A white pickup truck enters from the left side of the frame and proceeds straight.

Return = 3 timestamps for each white car.
"""
    assert extract_timestamps(test_text_2) == [
        (0, 0),
        (5, 5),
        (13, 13),
    ]

    # Option 3
    test_text_3 = """
Option 3: In the video, there are no black cars visible.

Return = 0 timestamps.
"""
    assert extract_timestamps(test_text_3) == []

    test_text_4 = """
- **Clip 1**:
    - **00:00:03 - 00:00:06**: A white Nissan NP200 pickup truck turns left from the street on the left side of the frame into the main road.
"""
    assert extract_timestamps(test_text_4) == [(3, 6)]

    # Tight range (no spaces) must still match
    assert extract_timestamps("See **0:01-0:02** here") == [(1, 2)]

    # Option 4 — multi-clip, spaced ranges, trailing ":" after bold (Untitled-1 lines 28–41)
    test_text_5 = """
*   **Clip 1 (0:00-0:05):**
    *   **0:01 - 0:02:** A white hatchback turns left from the left side road onto the main road.
    *   **0:02 - 0:03:** A silver SUV turns right from the main road onto the right side road.
    *   **0:04 - 0:05:** A white hatchback with a patterned roof turns right from the main road onto the right side road.

*   **Clip 2 (0:05-0:10):**
    *   **0:05 - 0:06:** A white truck with green crates in its bed turns left from the left side road onto the main road.
    *   **0:08 - 0:09:** A dark blue hatchback turns left from the left side road onto the main road.

*   **Clip 3 (0:10-0:15):**
    *   **0:11 - 0:12:** A silver hatchback turns left from the left side road onto the main road.
    *   **0:13 - 0:14:** A white pickup truck turns left from the left side road onto the main road.

=> Return 3 clip windows: (0, 5), (5, 10), (10, 15)
"""
    assert extract_timestamps(test_text_5) == [
        (0, 5),
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 10),
        (5, 6),
        (8, 9),
        (10, 15),
        (11, 12),
        (13, 14),
    ], extract_timestamps(test_text_5)

    # Full combined sample (Untitled-1) — all sections in one document
    test_text_full = """
Option 1: In this video, there are 8 turning vehicles.

**Clip 1 (0:00-0:15):**
- **0:01-0:02**: A white car turns right.
- **0:02-0:03**: A grey SUV turns right.
- **0:04-0:05**: A white truck with green crates turns left.
- **0:07-0:08**: A dark blue car turns left.
- **0:09-0:10**: A white car turns right.
- **0:10-0:11**: A silver car turns right.
- **0:13-0:14**: A white pickup truck turns right.
- **0:14-0:15**: A white car turns left.

Return = 1 timestamps (0, 15) (after Clip 1) for each turning vehicles.
==================================================
Option 2: There are **3** white cars in the video.

Here are the details:
1.  **00:00**: A white car (possibly a security vehicle with blue/purple markings) is seen turning left in the intersection.
2.  **00:05**: A white flatbed truck with green cages on the back enters from the left side of the frame and proceeds straight.
3.  **00:13**: A white pickup truck enters from the left side of the frame and proceeds straight.

=> Return 3 timestamps for each white car.
==================================================
Option 3: In the video, there are no black cars visible.

Return = 0 timestamps.
==================================================
*   **Clip 1 (0:00-0:05):**
    *   **0:01 - 0:02:** A white hatchback turns left from the left side road onto the main road.
    *   **0:02 - 0:03:** A silver SUV turns right from the main road onto the right side road.
    *   **0:04 - 0:05:** A white hatchback with a patterned roof turns right from the main road onto the right side road.

*   **Clip 2 (0:05-0:10):**
    *   **0:05 - 0:06:** A white truck with green crates in its bed turns left from the left side road onto the main road.
    *   **0:08 - 0:09:** A dark blue hatchback turns left from the left side road onto the main road.

*   **Clip 3 (0:10-0:15):**
    *   **0:11 - 0:12:** A silver hatchback turns left from the left side road onto the main road.
    *   **0:13 - 0:14:** A white pickup truck turns left from the left side road onto the main road.

=> Return 3 timestamps for each turning vehicles: (0, 5), (5, 10), (10, 15)
"""
    assert extract_timestamps(test_text_full) == [
        (0, 15),
        (1, 2),
        (2, 3),
        (4, 5),
        (7, 8),
        (9, 10),
        (10, 11),
        (13, 14),
        (14, 15),
        (0, 0),
        (5, 5),
        (13, 13),
        (0, 5),
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 10),
        (5, 6),
        (8, 9),
        (10, 15),
        (11, 12),
        (13, 14),
    ], extract_timestamps(test_text_full)

