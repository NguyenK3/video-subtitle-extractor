from enum import Enum


# Default approximate region where subtitles appear
class SubtitleArea(Enum):
    # Subtitle area appears in the lower part
    LOWER_PART = 0
    # Subtitle area appears in the upper part
    UPPER_PART = 1
    # Unknown subtitle area position
    UNKNOWN = 2
    # Custom subtitle area position
    CUSTOM = 3


class BackgroundColor(Enum):
    # Subtitle background
    WHITE = 0
    DARK = 1
    UNKNOWN = 2


BGR_COLOR_GREEN = (0, 0xff, 0)
BGR_COLOR_BLUE = (0xff, 0, 0)
BGR_COLOR_RED = (0, 0, 0xff)
BGR_COLOR_WHITE = (0xff, 0xff, 0xff)
