from typing import Any, Callable, Dict, Type

# The new unified, flat registry. Maps class names to class objects and default params.
PROTOCOL_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_protocol(default_params: Dict = None) -> Callable:
    """
    A decorator to register a protocol class in the central unified registry.
    The protocol is registered using its class name.
    """
    if default_params is None:
        default_params = {}

    def decorator(cls: Type) -> Type:
        name = cls.__name__
        if name in PROTOCOL_REGISTRY:
            raise TypeError(f"Protocol '{name}' is already registered.")

        PROTOCOL_REGISTRY[name] = {"class": cls, "default_params": default_params}
        return cls

    return decorator


def create_protocol(config: Dict, default_name: str = None) -> Any:
    """
    Factory function to recursively create protocol instances from a configuration dictionary.
    """
    if not config and default_name:
        config = {"name": default_name}
    elif not config and not default_name:
        # If no config and no default, we cannot proceed.
        raise ValueError("Cannot create protocol from an empty configuration without a default name.")

    # Fallback to default_name if 'name' is not in the config
    name = config.get("name", default_name)
    if not name:
        raise ValueError("Protocol name must be provided in config or as a default.")

    params = config.get("params", {})

    protocol_info = PROTOCOL_REGISTRY.get(name)
    if not protocol_info:
        raise ValueError(f"Protocol '{name}' not found in the registry.")

    protocol_class = protocol_info["class"]
    # Start with the protocol's defaults, then override with config params
    final_params = {**protocol_info["default_params"], **params}

    # --- Recursive Instantiation for Nested Protocols ---
    for param_name, param_value in final_params.items():
        # If a parameter's value is a dictionary that looks like a protocol config
        # (i.e., it has a "name" key), we recursively create it.
        if isinstance(param_value, dict) and "name" in param_value:
            # The nested protocol's config is the param_value itself.
            final_params[param_name] = create_protocol(param_value)

    return protocol_class(**final_params)
