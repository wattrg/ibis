import math
import numpy as np

class Unit:
    def __init__(self, dimension_list):
        assert(len(dimension_list) == 7)
        self._dim = dimension_list.copy()

    def __eq__(self, other):
        return self._dim == other._dim

    def __mul__(self, other):
        result = [0, 0, 0, 0, 0, 0, 0]
        for i, (self_dim, other_dim) in enumerate(zip(self._dim, other._dim)):
            result[i] = self_dim + other_dim
        return Unit(result)

    def __truediv__(self, other):
        result = [0, 0, 0, 0, 0, 0, 0]
        for i, (self_dim, other_dim) in enumerate(zip(self._dim, other._dim)):
            result[i] = self_dim - other_dim
        return Unit(result)

    def __add__(self, other):
        assert(self == other)
        return Unit(self._dim)

    def __sub__(self, other):
        assert(self == other)
        return Unit(self._dim)

    def __pow__(self, pow):
        result = [0, 0, 0, 0, 0, 0, 0]
        for i, unit in enumerate(self._dim):
            result[i] = unit*pow
        return Unit(result)

    def unit(self):
        return self._dim

class UnitVal:
    def __init__(self, value, unit):
        self._value = value
        self._unit = unit

    def __mul__(self, other):
        value = self._value * other._value
        unit = self._unit * other._unit
        return UnitVal(value, unit)

    def __truediv__(self, other):
        value = self._value / other._value
        unit = self._unit / other._unit
        return UnitVal(value, unit)

    def __add__(self, other):
        value = self._value + other._value
        unit = self._unit + other._unit
        return UnitVal(value, unit)

    def __sub__(self, other):
        value = self._value - other._value
        unit = self._unit - other._unit
        return UnitVal(value, unit)

    def __pow__(self, pow):
        value = self._value ** pow
        unit = self._unit ** pow
        return UnitVal(value, unit)
    
    def unit(self):
        return self._unit

    def value(self):
        return self._value

class ReferenceUnitSystem:
    def __init__(self, *args):
        if len(args) == 1:
            reference_values = args[0]
        else:
            reference_values = args

        if type(reference_values) == dict:
            self._from_dict(reference_values)
        else:
            self._from_reference_values(reference_values)


    def _from_reference_values(self, reference_values):
        x = self._solve_units(reference_values)
        self._time = x[0]
        self._length = x[1]
        self._mass = x[2]
        self._current = x[3]
        self._temp = x[4]
        self._amount = x[5]
        self._luminosity = x[6]


    def _from_dict(self, reference_values):
        self._time = reference_values["time"]
        self._length = reference_values["length"]
        self._mass = reference_values["mass"]
        self._current = reference_values["current"]
        self._temp = reference_values["temp"]
        self._amount = reference_values["amount"]
        self._luminosity = reference_values["luminosity"]

    def as_dict(self):
        return {
            "time": self._time,
            "length": self._length,
            "mass": self._mass,
            "current": self._current,
            "temp": self._temp,
            "amount": self._amount,
            "luminosity": self._luminosity
        }

    def time(self):
        return self._time

    def length(self):
        return self._length

    def velocity(self):
        return self.length() / self.time()

    def mass(self):
        return self._mass

    def current(self):
        return self._current

    def temp(self):
        return self._temp

    def viscosity(self):
        return self.length() * self.velocity()

    def density(self):
        return self.mass() / (self.length()**3)

    def pressure(self):
        return self.mass() / (self.length() * self.time()**2)

    def _count_units(self, reference_values):
        """
        Check the reference values provided are not over or 
        under constrained
        """
        included_units = []
        for reference_value in reference_values:
            units = reference_value.unit().unit()
            for i, unit in enumerate(units):
                if unit != 0 and i not in included_units:
                    included_units.append(i)
        included_units.sort()
        num_units = len(included_units)
        num_reference_values = len(reference_values)
        if num_reference_values < num_units:
            raise Exception("Under constrained system of reference units")
        if num_reference_values > num_units:
            raise Exception("Over constrained system of reference units")
        return included_units

    def _solve_units(self, reference_values):
        included_units = self._count_units(reference_values)
        n_units = len(included_units)
        matrix = np.zeros((n_units, n_units))
        for row in range(n_units):
            for col in range(n_units):
                unit_index = included_units[col]
                matrix[row,col] = reference_values[row].unit().unit()[unit_index]
        b = np.zeros(n_units)
        for i in range(n_units):
            b[i] = math.log10(reference_values[i].value())
        x_star = np.linalg.solve(matrix, b)
        x = [1.0] * 7
        for i, x_star_i in enumerate(x_star):
            x[included_units[i]] = 10**x_star_i
        return x

    def __str__(self):
        return (f"ReferenceUnitSystem(length={self.length()}, "
               f"mass={self.mass()}, "
               f"density={self.density()}, "
               f"time={self.time()}, "
               f"velocity={self.velocity()}, "
               f"temp={self.temp()}, "
               f"current={self.current()}, "
               f"viscosity={self.viscosity()})")

    def validate(self):
        pass

second   = Unit([1, 0, 0, 0, 0, 0, 0])
metre    = Unit([0, 1, 0, 0, 0, 0, 0])
kilogram = Unit([0, 0, 1, 0, 0, 0, 0])
ampere   = Unit([0, 0, 0, 1, 0, 0, 0])
kelvin   = Unit([0, 0, 0, 0, 1, 0, 0])
mole     = Unit([0, 0, 0, 0, 0, 1, 0])
candela  = Unit([0, 0, 0, 0, 0, 0, 1])

if __name__ == "__main__":
    length = UnitVal(0.5, metre)
    velocity = UnitVal(4, metre / second)
    density = UnitVal(0.1, kilogram / metre**3)
    temp = UnitVal(300, kelvin)

    ref_units = ReferenceUnitSystem([length, velocity, density, temp])
    print(ref_units)
