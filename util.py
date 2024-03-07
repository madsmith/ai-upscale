from dataclasses import dataclass
import json

class Timestamp:
    """Represents a timestamp in the format of the mkvextract timestamps_v2 output."""
    def __init__(self, value):
        self.value = int(float(value))

    def asTimestampString(self):
        hours = self.value // 3600000
        minute = (self.value // 60000) % 60
        second = (self.value // 1000) % 60
        millisecond = self.value % 1000

        return f"{hours:02d}:{minute:02d}:{second:02d}.{millisecond:03d}"

    def getValueStr(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, Timestamp):
            return self.value == other.value
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Timestamp):
            return self.value != other.value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Timestamp):
            return self.value < other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Timestamp):
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Timestamp):
            return self.value <= other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Timestamp):
            return self.value >= other.value
        return NotImplemented

    def __str__(self):
        return self.asTimestampString()

    def __fspath__(self):
        return self.asTimestampString()

    def __repr__(self):
        return f"Timestamp({self.value} - {self.asTimestampString()})"


@dataclass
class Resolution:
    width: int
    height: int

    def __getitem__(self, x):
        if x == 'w': return self.width
        if x == 'h': return self.height
        if x == 0: return self.width
        if x == 1: return self.height
        return None

    def __getattr__(self, name):
        if name == 'w':
            return self.width
        elif name == 'h':
            return self.height
        else:
            # Raise AttributeError to comply with Python's expected behavior
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __mul__(self, other):
        if isinstance(other, int):
            return Resolution(self.width * other, self.height * other)
        elif isinstance(other, float):
            raise Exception("Multiplication of Resolution by float not supported")
        elif isinstance(other, AspectRatio):
            # Generally, I'm assuming that height is well known and the width is the
            # dimension that needs to be scaled.  This may not result in the closest
            # match to the display aspect ratio but I don't want to introduce more
            # pixel interpolation than necessary.
            new_width = self.width * other.width // other.height
            # TODO: determine if we want to force route to an even value
            #new_width = self._round_even(self.width * other.width / other.height)
            return Resolution(new_width, self.height)

    def __rmul__(self, other):
        # This ensures multiplication works in both orders: scalar * Resolution and Resolution * scalar
        return self.__mul__(other)

    def _round_even(self, value):
        rounded_value = round(value)

        if rounded_value % 2 == 0:
            return rounded_value
        else:
            return rounded_value + 1 if value > rounded_value else rounded_value - 1

    def __str__(self):
        return f"{self.width}x{self.height}"

class AspectRatio:
    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self._divisor = self._gcd(self.width, self.height)

    @classmethod
    def from_string(cls, value):
        width, height = map(int, value.split(":"))
        return cls(width, height)

    @classmethod
    def from_resolution(cls, resolution):
        return cls(resolution.width, resolution.height)

    def _gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a

    def value(self):
        return self.width / self.height

    # Allow aspect ratios to be divided by each other
    def __truediv__(self, other):
        if isinstance(other, AspectRatio):
            return self.value() / other.value()
        return NotImplemented

    # Equality of two aspect ratios
    def __eq__(self, other):
        if isinstance(other, AspectRatio):
            return ((self.width//self._divisor) == (other.width//other._divisor) and
                    (self.height//self._divisor) == (other.height//other._divisor))
        return NotImplemented

    def __str__(self):
        str_value = f"{(self.width//self._divisor)}:{(self.height//self._divisor)}"
        if (len(str_value) > 5):
            str_value = f"{str_value} [{(self.width/self.height):.4f}]"
        return str_value

class Packet:
    def __init__(self, probe_data, time_base):
        self.data = probe_data
        # Convert time_base 1/1000 to a tuple (1, 1000)
        self.time_base = tuple(map(int, time_base.split("/")))

    def timestamp(self):
        # Prefer pkt_dts, then best_effort_timestamp, then pts
        if 'dts' in self.data:
            return self._timestamp_to_ms(self.data["dts"])
        if 'best_effort_timestamp' in self.data:
            return self._timestamp_to_ms(self.data["best_effort_timestamp"])
        if 'pts' in self.data:
            return self._timestamp_to_ms(self.data["pts"])
        return None

    def time(self):
        # Prefer pkt_dts, then best_effort_timestamp, then pts
        if 'dts_time' in self.data:
            return float(self.data["dts_time"])
        if 'best_effort_timestamp_time' in self.data:
            return float(self.data["best_effort_timestamp_time"])
        if 'pts_time' in self.data:
            return float(self.data["pts_time"])
        return None

    def _timestamp_to_ms(self, timestamp):
        return (timestamp * self.time_base[0]) / self.time_base[1] * 1000

    def media_type(self):
        return self.data.get("media_type")

    def stream_index(self):
        return self.data.get("stream_index")

    # Render
    def __str__(self):
        # Select best_effort_timestamp, pkt_dts, and pts and media_type and stream_index if present
        print_data = {k: self.data[k] for k in [
            "media_type", "stream_index",
            "pts", "pkt_dts", "best_effort_timestamp",
            "pts_time", "pkt_dts_time", "best_effort_timestamp_time"
            ] if k in self.data}
        return json.dumps(print_data, indent=2)