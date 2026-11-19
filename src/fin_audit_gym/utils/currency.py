"""Currency and minor unit mathematical utilities."""

import decimal
from decimal import Decimal


def convert_currency(
    amount_minor: int,
    rate: str,
    rounding_mode: str = "ROUND_HALF_UP",
) -> int:
    """Convert integer minor currency units using a string exchange rate.

    Args:
        amount_minor: Non-negative integer representing minor currency units.
        rate: Exchange rate formatted as a decimal string.
        rounding_mode: Rounding strategy from decimal module (default: ROUND_HALF_UP).

    Returns:
        Converted amount in integer minor currency units.

    Raises:
        ValueError: If amount_minor is negative, rate is invalid/non-positive,
                    or rounding mode is unrecognized.
    """
    if not isinstance(amount_minor, int) or amount_minor < 0:
        raise ValueError(f"amount_minor must be a non-negative integer, got {amount_minor}")

    try:
        dec_rate = Decimal(str(rate))
    except (decimal.InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid currency exchange rate '{rate}': {exc}") from exc

    if dec_rate <= Decimal(0):
        raise ValueError(f"Exchange rate must be strictly positive, got '{rate}'")

    rounding_attr = getattr(decimal, rounding_mode, None)
    if rounding_attr is None:
        raise ValueError(f"Unknown decimal rounding mode: '{rounding_mode}'")

    dec_amount = Decimal(amount_minor)
    result = dec_amount * dec_rate
    quantized = result.quantize(Decimal(1), rounding=rounding_attr)
    return int(quantized)
