from tenacity import retry, stop_after_attempt, wait_exponential


class AIServicesError(Exception):
    """Base exception for AIServices."""

    pass


class ProviderError(AIServicesError):
    """Exception raised for provider-related errors (e.g. API failures)."""

    pass


class ResourceNotFoundError(AIServicesError):
    """Exception raised when a requested resource (model, file) is not found."""

    pass


# Standard retry decorator for network/API calls
retry_api_call = retry(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
)
