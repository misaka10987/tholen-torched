from enum import Enum
from typing import Any

import numpy
import torch
from torch import Tensor
from torch.utils.data import Dataset


class SpecT(Enum):
    X = 0
    C = 1
    S = 2

    @classmethod
    def from_str(cls, name) -> "SpecT":
        string = name.upper()
        if string[0] in "CBFGDT":
            return cls.C
        elif string[0] in "SVAQR":
            return cls.S
        else:
            return cls.X

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> "SpecT":
        arr = tensor.detach().numpy()
        max_idx = numpy.argmax(arr)
        return max_idx

    def to_tensor(self) -> Tensor:
        return torch.tensor(self.value)


class OrbitClass(Enum):
    NULL = -1
    JFC = 0
    HTC = 1
    ETC = 2
    COM = 3
    HYP = 4
    CTC = 5
    PAR = 6
    MBA = 7
    APO = 8
    MCA = 9
    AMO = 10
    OMB = 11
    CEN = 12
    ATE = 13
    TNO = 14
    TJN = 15
    IMB = 16
    AST = 17
    IEO = 18
    HYA = 19

    @classmethod
    def from_str(cls, orbit_class: str) -> "OrbitClass":
        """Returns the OrbitClass enum value corresponding to the given string, regardless of case.
        """
        try:
            return cls[orbit_class.upper()]
        except KeyError:
            return cls.NULL


def replace_none(tuple_input):
    """Replaces all None values in a tuple with NaN.

    Args:
      tuple_input: The input tuple.

    Returns:
      A new tuple with all None values replaced with NaN.
    """

    # Create a new tuple with the same length as the input tuple.
    new_tuple = [None] * len(tuple_input)

    # Iterate over the input tuple and replace None values with NaN.
    for i, value in enumerate(tuple_input):
        if value is None:
            # noinspection PyTypeChecker
            new_tuple[i] = -1000
        else:
            new_tuple[i] = value

    # Return the new tuple.
    return tuple(new_tuple)


def opt(x) -> (Any, float):
    if x is None:
        return 0, 1.
    else:
        return x, 0.


class Asteroid:
    def __init__(self, bv: float, ub: float, ir: float, spec_t: str, a: float, condition_code: int, diameter: float,
                 e: float, i: float, m: float, n: float,
                 node: float, orbit_class: str, peri: float,
                 period: float, q: float,
                 qq: float,
                 rot_per: float):
        self.bv: float = bv
        self.ub: float = ub
        self.ir: float = ir
        self.spec_t: SpecT = SpecT.from_str(spec_t)
        self.a: float = a
        self.condition_code: int = condition_code
        self.diameter: float = diameter
        self.e: float = e
        self.i: float = i
        self.m: float = m
        self.n: float = n
        self.node: float = node
        self.orbit_class: OrbitClass = OrbitClass.from_str(orbit_class)
        self.peri: float = peri
        self.period: float = period
        self.q: float = q
        self.qq: float = qq
        self.rot_per: float = rot_per

    @classmethod
    def from_row(cls, t: tuple):
        return Asteroid(*t[1:])

    def to_tensor(self) -> Tensor:
        tup = (
            self.bv, self.ub, self.ir, self.a, float(self.condition_code), self.diameter, self.e, self.i, self.m,
            self.n,
            self.node, float(self.orbit_class.value), self.peri, self.period, self.q, self.qq, self.rot_per
        )
        return torch.tensor(replace_none(tup), dtype=torch.float32)

    def to_label(self) -> Tensor:
        return self.spec_t.to_tensor()


class MyData(Dataset):
    def __init__(self, data_list: list[Asteroid]):
        self.data_list: list[Asteroid] = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx) -> Asteroid:
        return self.data_list[idx]
