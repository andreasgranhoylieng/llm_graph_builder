"""
RateLimiter Service - Handles API rate limiting with token bucket algorithm.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Optional
from src import config


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 500
    tokens_per_minute: int = 150000
    burst_allowance: float = 1.2  # Allow 20% burst


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Thread-safe and async-compatible.
    """

    def __init__(self, requests_per_minute: int = None, tokens_per_minute: int = None):
        self.rpm = requests_per_minute or config.RATE_LIMIT_RPM
        self.tpm = tokens_per_minute or config.RATE_LIMIT_TPM

        # Token buckets
        self._request_tokens = self.rpm
        self._token_tokens = self.tpm

        # Refill rates (per second)
        self._request_refill_rate = self.rpm / 60.0
        self._token_refill_rate = self.tpm / 60.0

        # Timing
        self._last_refill = time.time()

        # Thread safety
        self._lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0.0

    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._last_refill = now

        # Refill request tokens
        self._request_tokens = min(
            self.rpm, self._request_tokens + (elapsed * self._request_refill_rate)
        )

        # Refill token tokens
        self._token_tokens = min(
            self.tpm, self._token_tokens + (elapsed * self._token_refill_rate)
        )

    def acquire(self, estimated_tokens: int = 1000) -> float:
        """
        Acquire permission to make an API call.
        Blocks until rate limit allows.

        Args:
            estimated_tokens: Estimated token usage for this request

        Returns:
            Time spent waiting (seconds)
        """
        wait_time = 0.0

        with self._lock:
            while True:
                self._refill_tokens()

                # Check if we have enough tokens
                if self._request_tokens >= 1 and self._token_tokens >= estimated_tokens:
                    self._request_tokens -= 1
                    self._token_tokens -= estimated_tokens
                    self.total_requests += 1
                    self.total_tokens += estimated_tokens
                    self.total_wait_time += wait_time
                    return wait_time

                # Calculate wait time
                request_wait = (
                    0
                    if self._request_tokens >= 1
                    else (1 - self._request_tokens) / self._request_refill_rate
                )
                token_wait = (
                    0
                    if self._token_tokens >= estimated_tokens
                    else (estimated_tokens - self._token_tokens)
                    / self._token_refill_rate
                )

                sleep_time = max(request_wait, token_wait, 0.1)

                # Release lock while sleeping
                self._lock.release()
                try:
                    time.sleep(min(sleep_time, 1.0))  # Sleep max 1 second at a time
                    wait_time += min(sleep_time, 1.0)
                finally:
                    self._lock.acquire()

    async def acquire_async(self, estimated_tokens: int = 1000) -> float:
        """
        Async version of acquire for use with asyncio.
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        wait_time = 0.0

        async with self._async_lock:
            while True:
                self._refill_tokens()

                if self._request_tokens >= 1 and self._token_tokens >= estimated_tokens:
                    self._request_tokens -= 1
                    self._token_tokens -= estimated_tokens
                    self.total_requests += 1
                    self.total_tokens += estimated_tokens
                    self.total_wait_time += wait_time
                    return wait_time

                request_wait = (
                    0
                    if self._request_tokens >= 1
                    else (1 - self._request_tokens) / self._request_refill_rate
                )
                token_wait = (
                    0
                    if self._token_tokens >= estimated_tokens
                    else (estimated_tokens - self._token_tokens)
                    / self._token_refill_rate
                )

                sleep_time = max(request_wait, token_wait, 0.1)
                await asyncio.sleep(min(sleep_time, 1.0))
                wait_time += min(sleep_time, 1.0)

    def report_actual_tokens(self, actual_tokens: int, estimated_tokens: int):
        """
        Adjust token bucket based on actual usage.
        Call this after receiving API response with actual token count.
        """
        difference = estimated_tokens - actual_tokens
        with self._lock:
            # Credit back overestimated tokens
            self._token_tokens = min(self.tpm, self._token_tokens + difference)

    def handle_rate_limit_error(self, retry_after: float = None):
        """
        Handle a 429 rate limit error from the API.
        Reduces available tokens and returns suggested wait time.
        """
        with self._lock:
            # Drain tokens to prevent immediate retry
            self._request_tokens = 0
            self._token_tokens = 0
            self._last_refill = time.time()

        suggested_wait = retry_after or 60.0
        print(f"⚠️ Rate limited! Waiting {suggested_wait:.1f}s before retrying...")
        return suggested_wait

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "current_request_tokens": round(self._request_tokens, 2),
            "current_token_tokens": round(self._token_tokens, 0),
        }


# Global rate limiter instance (singleton pattern)
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def reset_rate_limiter():
    """Reset the global rate limiter (useful for testing)."""
    global _global_rate_limiter
    _global_rate_limiter = None
