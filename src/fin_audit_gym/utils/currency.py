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
    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int) or amount_minor < 0:
        raise ValueError(f"amount_minor must be a non-negative integer, got {amount_minor}")

    try:
        dec_rate = Decimal(str(rate))
        if dec_rate.is_nan() or dec_rate.is_infinite() or dec_rate <= Decimal(0):
            raise ValueError(
                f"Exchange rate must be a finite, strictly positive number, got '{rate}'"
            )
    except (decimal.InvalidOperation, TypeError) as exc:
        raise ValueError(f"Invalid currency exchange rate '{rate}': {exc}") from exc

    valid_rounding_modes = {
        "ROUND_HALF_UP",
        "ROUND_HALF_EVEN",
        "ROUND_HALF_DOWN",
        "ROUND_UP",
        "ROUND_DOWN",
        "ROUND_CEILING",
        "ROUND_FLOOR",
        "ROUND_05UP",
    }
    if rounding_mode not in valid_rounding_modes:
        raise ValueError(f"Unknown decimal rounding mode: '{rounding_mode}'")

    rounding_attr = getattr(decimal, rounding_mode)

    dec_amount = Decimal(amount_minor)
    result = dec_amount * dec_rate
    quantized = result.quantize(Decimal(1), rounding=rounding_attr)
    return int(quantized)
