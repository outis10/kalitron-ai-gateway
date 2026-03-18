class AIGatewayError(Exception):
    """Base exception for controlled gateway failures."""


class RuntimeConfigurationError(AIGatewayError):
    """Raised when required runtime configuration is missing or invalid."""


class UpstreamServiceError(AIGatewayError):
    """Raised when an upstream provider cannot be reached reliably."""


class ProviderResponseError(AIGatewayError):
    """Raised when an upstream provider returns an invalid response shape."""
