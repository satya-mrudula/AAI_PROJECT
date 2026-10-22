"""Cost calculator for provider-reported token usage."""


def calculate_cost(input_tokens: int, output_tokens: int, input_price_per_million: float, output_price_per_million: float) -> float:
    values = (input_tokens, output_tokens, input_price_per_million, output_price_per_million)
    if any(value < 0 for value in values):
        raise ValueError("token counts and token prices must be non-negative")
    return (input_tokens / 1_000_000 * input_price_per_million) + (output_tokens / 1_000_000 * output_price_per_million)
