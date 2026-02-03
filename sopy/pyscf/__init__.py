try:
    from .ext import get_orbital
    from .ext import get_basis
    from .ext import tabulate
    from .ext import rotation_matrix_from_euler
    
    __all__ = ['get_orbital', 'get_basis', 'tabulate', 'rotation_matrix_from_euler']

except ImportError as e:
    if 'pyscf' in str(e):
        raise ImportError(
            "Missing optional dependency 'pyscf'. "
            "Please install it using: pip install sopy[pyscf]"
        ) from e
    else:
        raise e