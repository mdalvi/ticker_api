from __future__ import annotations


class TickerTapeException(Exception):
    def __init__(self, message: str | None = None):
        self.message = message


class MultipleTokensFoundException(TickerTapeException):
    """
    Occurs when multiple instrument_tokens are found against a (tradingsymbol, exchange) pair
    """

    def __init__(self, message: str | None = None):
        super().__init__(message)


class InvalidExchangeException(TickerTapeException):
    """
    Occurs when exchange is not supported for operations by TickerClass
    """

    def __init__(self, message: str | None = None):
        super().__init__(message)


class InvalidSegmentException(TickerTapeException):
    """
    Occurs when segment is not supported for operations by TickerClass
    """

    def __init__(self, message: str | None = None):
        super().__init__(message)
