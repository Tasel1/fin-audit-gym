"""Custom exceptions for the Fin-Audit-Gym environment."""


class FinAuditGymError(Exception):
    """Base exception for all Fin-Audit-Gym environment errors."""


class InvalidActionError(FinAuditGymError):
    """Raised when an agent provides an unrecognized or malformed action."""


class EpisodeDoneError(FinAuditGymError):
    """Raised when step() is called on an already terminated or truncated episode."""
