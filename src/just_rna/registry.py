"""Explicit clock registry without import-time mutation."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from types import MappingProxyType
from typing import cast

from just_rna.clocks.tage.clock import TAgeClock
from just_rna.exceptions import UnknownClockError
from just_rna.models import ClockId

ClockFactory = Callable[[], object]

CLOCK_FACTORIES: Mapping[ClockId, ClockFactory] = MappingProxyType(
    {
        ClockId.TAGE: TAgeClock,
    }
)


def get_clock(clock: ClockId) -> TAgeClock:
    """Construct a registered clock.

    The return type is narrowed to the only currently registered implementation.
    It becomes a discriminated overload when additional clocks are added.
    """

    factory = CLOCK_FACTORIES.get(clock)
    if factory is None:
        raise UnknownClockError(clock.value)
    return cast(TAgeClock, factory())


__all__ = ["CLOCK_FACTORIES", "get_clock"]

