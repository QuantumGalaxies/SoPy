try:
    from .ext import reduce
    from .ext import image
    __all__ = ['reduce', 'image']

except ImportError as e:
    # Check if the error is actually because 'tensorly' is missing
    # (and not some other syntax error in your code)
    if 'tensorly' in str(e) or 'tensorflow' in str(e):
        raise ImportError(
            "Missing optional dependency 'tensorly'. "
            "Please install it using: pip install sopy[tensorly]"
        ) from e
    else:
        raise e