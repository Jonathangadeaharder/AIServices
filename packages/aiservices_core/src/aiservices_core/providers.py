from abc import ABC, abstractmethod
from typing import Any
from .config import config


class BaseProvider(ABC):
    """Abstract Base Class for all AI service providers."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def generate(self, request: Any, output_path: str | None = None) -> Any:
        """Generate output from a request."""
        pass


class MemoryManagedProvider(BaseProvider):
    """Base class for memory managed providers to support isinstance checks."""
    pass


def wrap_provider(provider: BaseProvider) -> MemoryManagedProvider:
    provider_cls = provider.__class__

    # Dynamically subclass both MemoryManagedProvider and the original provider class
    class MemoryManagedProviderImpl(MemoryManagedProvider, provider_cls):
        def __init__(self, provider: BaseProvider):
            # Set the wrapped provider instance without triggering __setattr__ delegation
            object.__setattr__(self, "_provider", provider)
            
            # Apply hardware limits/memory guards on instantiation
            from .runtime import setup_memory_guards
            setup_memory_guards()

        def generate(self, request: Any, output_path: str | None = None) -> Any:
            try:
                return self._provider.generate(request, output_path)
            finally:
                if config.auto_cleanup:
                    from .runtime import cleanup_memory
                    cleanup_memory()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._provider, name)

        def __setattr__(self, name: str, value: Any):
            if name == "_provider":
                object.__setattr__(self, name, value)
            else:
                if hasattr(self, "_provider"):
                    setattr(self._provider, name, value)
                else:
                    object.__setattr__(self, name, value)

        def __dir__(self) -> list[str]:
            return list(set(super().__dir__() + dir(self._provider)))

    # Clean class representation
    MemoryManagedProviderImpl.__name__ = "MemoryManagedProvider"
    MemoryManagedProviderImpl.__qualname__ = "MemoryManagedProvider"
    return MemoryManagedProviderImpl(provider)


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
        raw_provider = self._providers[name](**kwargs)
        return wrap_provider(raw_provider)


registry = ProviderRegistry()

