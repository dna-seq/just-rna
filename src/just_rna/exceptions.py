"""Typed failures exposed by the public just-rna API."""

from __future__ import annotations


class JustRnaError(Exception):
    """Base class for expected just-rna failures."""


class InvalidExpressionError(JustRnaError, ValueError):
    """An expression matrix violates the declared data contract."""


class IncompatibleExpressionScaleError(InvalidExpressionError):
    """A clock cannot consume the declared expression scale."""


class MissingModelFeaturesError(InvalidExpressionError):
    """Expression data does not contain features required by a model."""


class SampleAlignmentError(InvalidExpressionError):
    """Expression columns and metadata sample identifiers do not align."""


class UnknownClockError(JustRnaError, KeyError):
    """A requested clock is not registered."""


class UnknownModelError(JustRnaError, KeyError):
    """A requested model is not registered."""

