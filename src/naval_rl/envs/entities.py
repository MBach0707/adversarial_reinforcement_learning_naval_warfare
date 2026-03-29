"""
Naval entity classes: Ship, Weapon, ADMeasures.

All positional units are metres; speed is in knots internally converted
to m/min on movement; course is in radians (mathematical convention,
0 = East, CCW positive).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utility conversions
# ---------------------------------------------------------------------------

def knots_to_mpm(knots: float) -> float:
    """Convert knots → metres per minute."""
    return knots * 1852.0 / 60.0


def mpm_to_knots(mpm: float) -> float:
    """Convert metres per minute → knots."""
    return mpm * 60.0 / 1852.0


def course_from_north(rad: float) -> float:
    """Convert math-convention radians to compass bearing (degrees, 0=N, CW)."""
    return (90.0 - math.degrees(rad)) % 360.0


# ---------------------------------------------------------------------------
# Weapon
# ---------------------------------------------------------------------------

@dataclass
class Weapon:
    """Surface-to-surface missile / gun system."""

    name: str
    stockpile: int
    range: float          # metres
    cooldown_max: int     # timesteps (minutes)
    p_hit: float          # probability of hitting per missile
    max_salvo_size: int

    stockpile_max: int = field(init=False)
    cooldown: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.stockpile_max = self.stockpile

    # ------------------------------------------------------------------
    def can_fire(self) -> bool:
        return self.stockpile > 0 and self.cooldown == 0

    def fire(self, salvo_size: int) -> int:
        """Fire *salvo_size* missiles (clamped to stockpile). Returns shots fired."""
        shots = min(salvo_size, self.stockpile)
        self.stockpile -= shots
        self.cooldown = self.cooldown_max
        return shots

    def tick(self) -> None:
        """Advance cooldown by one timestep."""
        self.cooldown = max(0, self.cooldown - 1)

    def reset(self) -> None:
        self.stockpile = self.stockpile_max
        self.cooldown = 0


# ---------------------------------------------------------------------------
# Air-Defence Measures
# ---------------------------------------------------------------------------

@dataclass
class ADMeasure:
    """Surface-to-air missile / point-defence system."""

    name: str
    stockpile: int
    range: float
    cooldown_max: int

    stockpile_max: int = field(init=False)
    cooldown: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.stockpile_max = self.stockpile

    # ------------------------------------------------------------------
    def can_intercept(self) -> bool:
        return self.stockpile > 0 and self.cooldown == 0

    def intercept(self, salvo_size: int) -> int:
        """Attempt to intercept *salvo_size* incoming missiles."""
        intercepted = min(salvo_size, self.stockpile)
        self.stockpile -= intercepted
        self.cooldown = self.cooldown_max
        return intercepted

    def tick(self) -> None:
        self.cooldown = max(0, self.cooldown - 1)

    def reset(self) -> None:
        self.stockpile = self.stockpile_max
        self.cooldown = 0


# ---------------------------------------------------------------------------
# Ship
# ---------------------------------------------------------------------------

ShotRecord = dict  # {"source", "target", "missiles_fired", "intercepted", "missiles_hit"}


class Ship:
    """
    A surface combatant represented as a point agent on a 2-D plane.

    Parameters
    ----------
    name        : display name
    x, y        : starting position (metres)
    course      : starting course (radians, mathematical convention)
    max_speed   : maximum speed (knots)
    health      : hit points
    grid_size   : (half-width, half-height) of the operational area (metres)
    history_len : rolling window length for course / speed averaging
    """

    def __init__(
        self,
        name: str,
        x: float,
        y: float,
        course: float = 0.0,
        max_speed: float = 20.0,
        health: int = 5,
        grid_size: Tuple[float, float] = (100_000.0, 100_000.0),
        history_len: int = 10,
    ) -> None:
        self.name = name
        self.grid_size = np.asarray(grid_size, dtype=np.float32)

        # Starting config (saved for reset)
        self._start = (float(x), float(y), float(course), int(health), 0.0)

        # Dynamic state
        self.x = float(x)
        self.y = float(y)
        self.course = float(course)
        self.speed: float = 0.0          # m/min (current)
        self.max_speed = float(max_speed)
        self.health = int(health)
        self.max_health = int(health)
        self.alive = True
        self.kill_count = 0

        # Equipment
        self.weapons: List[Weapon] = []
        self.ad_measures: List[ADMeasure] = []

        # Per-step shot log (filled by environment)
        self.last_shots: List[ShotRecord] = []

        # Rolling history for visualisation
        self.course_history: deque = deque([float(course)], maxlen=history_len)
        self.speed_history: deque = deque([0.0], maxlen=history_len)

    # ------------------------------------------------------------------
    # Equipment
    # ------------------------------------------------------------------

    def add_weapons(self, weapons: List[Weapon]) -> None:
        self.weapons.extend(weapons)

    def add_ad_measures(self, measures: List[ADMeasure]) -> None:
        self.ad_measures.extend(measures)

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def move(self, course: float, speed_fraction: float) -> None:
        """
        Apply a movement command.

        Parameters
        ----------
        course         : desired course (radians)
        speed_fraction : fraction of max speed [0, 1]
        """
        if not self.alive:
            return

        self.course = float(course)
        self.speed = float(np.clip(speed_fraction, 0.0, 1.0)) * knots_to_mpm(self.max_speed)

        dx = self.speed * math.cos(self.course)
        dy = self.speed * math.sin(self.course)
        self.x += dx
        self.y += dy

        self.course_history.append(self.course)
        self.speed_history.append(self.speed)

    # ------------------------------------------------------------------
    # Damage / lifecycle
    # ------------------------------------------------------------------

    def take_damage(self, amount: int = 1) -> None:
        self.health -= amount
        if self.health <= 0:
            self.alive = False

    def is_alive(self) -> bool:
        return self.alive

    # ------------------------------------------------------------------
    # Config serialisation (used by env reset)
    # ------------------------------------------------------------------

    def get_config(self) -> tuple:
        return (self.x, self.y, self.course, self.health, self.speed)

    def set_config(self, cfg: tuple) -> None:
        self.x, self.y, self.course, self.health, self.speed = cfg
        self.alive = self.health > 0

    def reset(self) -> None:
        self.set_config(self._start)
        self.alive = True
        self.kill_count = 0
        self.last_shots = []
        self.course_history = deque([self._start[2]], maxlen=self.course_history.maxlen)
        self.speed_history = deque([0.0], maxlen=self.speed_history.maxlen)
        for w in self.weapons:
            w.reset()
        for ad in self.ad_measures:
            ad.reset()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_pos(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def avg_course(self) -> float:
        return float(np.mean(self.course_history))

    @property
    def avg_speed_knots(self) -> float:
        return mpm_to_knots(float(np.mean(self.speed_history)))

    def __repr__(self) -> str:
        return (
            f"Ship({self.name!r}, pos=({self.x:.0f},{self.y:.0f}), "
            f"hp={self.health}/{self.max_health}, alive={self.alive})"
        )
