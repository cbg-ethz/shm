import enum


class Enum(enum.Enum):

    @classmethod
    def from_str(cls, value):
        for key, item in cls.__members__.items():
            if value == key:
                return item
        raise KeyError("Enum does not have value: {}".format(value))
