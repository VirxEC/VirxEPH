import json
from pathlib import Path

import numpy as np
from rlbot.utils.structures.game_data_struct import GameTickPacket, PlayerInfo, BallInfo

NAMES = (
    "may_ground_shot",
    "may_jump_shot",
    "may_double_jump_shot",
    "may_aerial"
)
DEFAULTS = [1, 1, 1, 1]


class PacketHeuristicAnalyzer:
    def __init__(self, threshold=0.8, gain=0.21, loss=0.005, unpause_delay=1.5, ignore_indexes=[]):
        self.cars = {}
        self.car_tracker = {}

        self.threshold = threshold
        self.gain = gain
        self.loss = loss

        self.ignore_indexes = ignore_indexes
        self.time = 0
        self.last_ball_touch_time = -1
        self.unpause_delay = unpause_delay
        self.last_pause_time = -1
        self.team_count = [0, 0]

    def add_packet(self, packet: GameTickPacket) -> bool:
        time = packet.game_info.seconds_elapsed
        delta_time = time - self.time
        self.time = time

        if not packet.game_info.is_round_active:
            self.last_pause_time = self.time
            return False

        if self.time - self.last_pause_time < self.unpause_delay:
            return False

        team_count = [0, 0]

        for i in range(packet.num_cars):
            team_count[packet.game_cars[i].team] += 1

        self.team_count = team_count

        loss = self.loss * delta_time

        latest_touch = packet.game_ball.latest_touch
        handled_touch = latest_touch.time_seconds != self.last_ball_touch_time
        self.last_ball_touch_time = latest_touch.time_seconds

        for i in range(packet.num_cars):
            if i in self.ignore_indexes:
                continue

            car = packet.game_cars[i]
            if car.is_demolished:
                continue

            if car.name not in self.cars:
                self.cars[car.name] = {}

            friends = team_count[car.team] - 1
            foes = team_count[car.team]

            if friends not in self.cars[car.name]:
                self.cars[car.name][friends] = {}

            if foes not in self.cars[car.name][friends]:
                self.cars[car.name][friends][foes] = DEFAULTS.copy()

            if car.name not in self.car_tracker:
                self.car_tracker[car.name] = {
                    "last_wheel_contact": {
                        "time": -1,
                        "up": Vector(),
                        "location": Vector()
                    }
                }

            if car.has_wheel_contact:
                self.car_tracker[car.name]['last_wheel_contact']['time'] = self.time
                self.car_tracker[car.name]['last_wheel_contact']['location'] = Vector.from_vector(car.physics.location)
                CP = math.cos(car.physics.rotation.pitch)
                SP = math.sin(car.physics.rotation.pitch)
                CY = math.cos(car.physics.rotation.yaw)
                SY = math.sin(car.physics.rotation.yaw)
                CR = math.cos(car.physics.rotation.roll)
                SR = math.sin(car.physics.rotation.roll)
                self.car_tracker[car.name]['last_wheel_contact']['up'] = Vector(-CR*CY*SP-SR*SY, -CR*SY*SP+SR*CY, CP*CR)

            # Ball heuristic
            ball_section = self.get_ball_section(packet.game_ball, car)
            self.cars[car.name][friends][foes][ball_section] = max(self.cars[car.name][friends][foes][ball_section] - (loss / (friends+1)), 0)

            if not handled_touch and latest_touch.player_index == i:
                time_airborne = self.time - self.car_tracker[car.name]['last_wheel_contact']['time']
                # REVISE TO USE self.car_tracker[car.name]["last_wheel_contact"]
                divisors = [
                    car.has_wheel_contact,
                    ball_section in {1, 2} and car.jumped and not car.double_jumped,
                    ball_section in {1, 2, 3} and car.jumped and car.double_jumped,
                    ball_section in {2, 3} and (time_airborne > 0.75 or not car.jumped),
                    True  # We're just going to ignore this touch
                ]
                ball_touch_section = divisors.index(True)
                if index != 5:
                    self.cars[car.name][friends][foes][ball_touch_section] = min(self.cars[car.name][friends][foes][ball_touch_section] + self.gain + loss, 1)
            
        return True
    
    def get_ball_section(self, ball: BallInfo, car: PlayerInfo) -> int:
        location = Vector.from_vector(ball.physics.location) - self.car_tracker[car.name]['last_wheel_contact']['location']
        
        dbz = self.car_tracker[car.name]['last_wheel_contact']['location'].dot(location)
        divisors = [
            dbz <= 126.75,
            dbz <= 312.75,
            dbz <= 542.75,
            True
        ]

        return divisors.index(True)

    def get_car(self, car_name: str, car_team: int) -> list:
        if car_name not in self.cars:
            return None

        return self.cars[car_name][self.team_count[car_team] - 1][self.team_count[not car_team]]

    def predict_car(self, car: list):
        return {NAMES[i]: car[i] > self.threshold for i in range(len(DEFAULTS))}


# Vector supports 1D, 2D and 3D Vectors, as well as calculations between them
# Arithmetic with 1D and 2D lists/tuples aren't supported - just set the remaining values to 0 manually
# With this new setup, Vector is much faster because it's just a wrapper for numpy
class Vector:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, np_arr=None):
        # this is a private property - this is so all other things treat this class like a list, and so should you!
        self._np = np.array([x, y, z]) if np_arr is None else np_arr

    def __getitem__(self, index):
        return self._np[index].item()

    def __setitem__(self, index, value):
        self._np[index] = value

    @property
    def x(self):
        return self._np[0].item()

    @x.setter
    def x(self, value):
        self._np[0] = value

    @property
    def y(self):
        return self._np[1].item()

    @y.setter
    def y(self, value):
        self._np[1] = value

    @property
    def z(self):
        return self._np[2].item()

    @z.setter
    def z(self, value):
        self._np[2] = value

    # self == value
    def __eq__(self, value):
        if isinstance(value, float) or isinstance(value, int):
            return self.magnitude() == value

        if hasattr(value, "_np"):
            value = value._np
        return (self._np == value).all()

    # len(self)
    def __len__(self):
        return 3  # this is a 3 dimensional vector, so we return 3

    # str(self)
    def __str__(self):
        # Vector's can be printed to console
        return f"[{self.x} {self.y} {self.z}]"

    # repr(self)
    def __repr__(self):
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"

    # -self
    def __neg__(self):
        return Vector(np_arr=self._np * -1)

    # self + value
    def __add__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np+value)
    __radd__ = __add__

    # self - value
    def __sub__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np-value)

    def __rsub__(self, value):
        return -self + value

    # self * value
    def __mul__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np*value)
    __rmul__ = __mul__

    # self / value
    def __truediv__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np/value)

    def __rtruediv__(self, value):
        return self * (1 / value)

    # round(self)
    def __round__(self, decimals=0) -> Vector:
        # Rounds all of the values
        return Vector(np_arr=np.around(self._np, decimals=decimals))

    @staticmethod
    def from_vector(vec) -> Vector:
        return Vector(vec.x, vec.y, vec.z)

    def magnitude(self) -> float:
        # Returns the length of the vector
        return np.linalg.norm(self._np).item()

    def _magnitude(self) -> np.float64:
        # Returns the length of the vector in a numpy float 64
        return np.linalg.norm(self._np)

    def dot(self, value: Vector) -> float:
        # Returns the dot product of two vectors
        if hasattr(value, "_np"):
            value = value._np
        return self._np.dot(value).item()

    def cross(self, value: Vector) -> Vector:
        # Returns the cross product of two vectors
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=np.cross(self._np, value))

    def copy(self) -> Vector:
        # Returns a copy of the vector
        return Vector(*self._np)

    def normalize(self, return_magnitude=False) -> List[Vector, float] or Vector:
        # normalize() returns a Vector that shares the same direction but has a length of 1
        # normalize(True) can also be used if you'd like the length of this Vector (used for optimization)
        magnitude = self._magnitude()
        if magnitude != 0:
            norm_vec = Vector(np_arr=self._np / magnitude)
            if return_magnitude:
                return norm_vec, magnitude.item()
            return norm_vec
        if return_magnitude:
            return Vector(), 0
        return Vector()

    def _normalize(self) -> np.ndarray:
        # Normalizes a Vector and returns a numpy array
        magnitude = self._magnitude()
        if magnitude != 0:
            return self._np / magnitude
        return np.array((0, 0, 0))

    def flatten(self) -> Vector:
        # Sets Z (Vector[2]) to 0, making the Vector 2D
        return Vector(self._np[0], self._np[1])

    def angle2D(self, value: Vector) -> float:
        # Returns the 2D angle between this Vector and another Vector in radians
        return self.flatten().angle(value.flatten())

    def angle(self, value: Vector) -> float:
        # Returns the angle between this Vector and another Vector in radians
        dp = np.dot(self._normalize(), value._normalize()).item()
        return math.acos(-1 if dp < -1 else (1 if dp > 1 else dp))

    def rotate2D(self, angle: float) -> Vector:
        # Rotates this Vector by the given angle in radians
        # Note that this is only 2D, in the x and y axis
        return Vector((math.cos(angle)*self.x) - (math.sin(angle)*self.y), (math.sin(angle)*self.x) + (math.cos(angle)*self.y), self.z)

    def clamp2D(self, start: Vector, end: Vector) -> Vector:
        # Similar to integer clamping, Vector's clamp2D() forces the Vector's direction between a start and end Vector
        # Such that Start < Vector < End in terms of clockwise rotation
        # Note that this is only 2D, in the x and y axis
        s = self._normalize()
        right = np.dot(s, np.cross(end._np, (0, 0, -1))) < 0
        left = np.dot(s, np.cross(start._np, (0, 0, -1))) > 0
        if (right and left) if np.dot(end._np, np.cross(start._np, (0, 0, -1))) > 0 else (right or left):
            return self
        if np.dot(start._np, s) < np.dot(end._np, s):
            return end
        return start

    def clamp(self, start: Vector, end: Vector) -> Vector:
        # This extends clamp2D so it also clamps the vector's z
        s = self.clamp2D(start, end)

        if s.z < start.z:
            s = s.flatten().scale(1 - start.z)
            s.z = start.z
        elif s.z > end.z:
            s = s.flatten().scale(1 - end.z)
            s.z = end.z

        return s

    def dist(self, value: Vector) -> float:
        # Distance between 2 vectors
        if hasattr(value, "_np"):
            value = value._np
        return np.linalg.norm(self._np - value).item()

    def flat_dist(self, value: Vector) -> float:
        # Distance between 2 vectors on a 2D plane
        return value.flatten().dist(self.flatten())

    def cap(self, low: float, high: float) -> Vector:
        # Caps all values in a Vector between 'low' and 'high'
        return Vector(*(max(min(item, high), low) for item in self._np))

    def midpoint(self, value: Vector) -> Vector:
        # Midpoint of the 2 vectors
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=(self._np + value) / 2)

    def scale(self, value: float) -> Vector:
        # Returns a vector that has the same direction but with a value as the magnitude
        return self.normalize() * value
