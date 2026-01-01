"""
PyTorch version compatibility utilities
"""

import torch


def get_custom_fwd_decorator():
    """
    Get the appropriate custom_fwd decorator based on PyTorch version.

    Returns:
        A decorator function that works with both PyTorch 1.x and 2.x
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                # Try PyTorch 2.x style first
                with torch.cuda.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32):
                    return func(*args, **kwargs)
            except (TypeError, AttributeError):
                # Fall back to PyTorch 1.x style
                with torch.cuda.amp.custom_fwd(cast_inputs=torch.float32):
                    return func(*args, **kwargs)

        return wrapper

    return decorator


# Alternative: direct decorator for method decoration
def custom_fwd_compatible(func):
    """
    A decorator that applies custom_fwd with version compatibility.

    Usage:
        @custom_fwd_compatible
        def my_method(self, ...):
            ...
    """

    def wrapper(*args, **kwargs):
        try:
            # Try PyTorch 2.x style first
            with torch.cuda.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32):
                return func(*args, **kwargs)
        except (TypeError, AttributeError):
            # Fall back to PyTorch 1.x style
            with torch.cuda.amp.custom_fwd(cast_inputs=torch.float32):
                return func(*args, **kwargs)

    return wrapper
