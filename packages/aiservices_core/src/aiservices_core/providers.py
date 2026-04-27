from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Abstract Base Class for all AI service providers."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass


class ProviderRegistry:
    """Registry to manage and instantiate providers."""

    def __init__(self):
        self._providers: dict[str, type[BaseProvider]] = {}

    def register(self, name: str, provider_cls: type[BaseProvider]):
        self._providers[name] = provider_cls

    def get(self, name: str, **kwargs) -> BaseProvider:
        if name not in self._providers:
            available = list(self._providers.keys())
            raise ValueError(f"Provider '{name}' not found. Available: {available}")
        return self._providers[name](**kwargs)


registry = ProviderRegistry()
