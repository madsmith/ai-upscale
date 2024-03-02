
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
