# A generic stub that covers all the required extras for this module
def _missing_ext_stub(*args, **kwargs):
    raise ImportError(
        "This feature requires additional optional dependencies (pyscf, numpy, pandas). "
        "Please install them using: pip install sopy-quantum[pyscf]" 
        # Update the bracket tag above if you named the group something else!
    )

try:
    from .ext import reduce, image

except ImportError as e:
    # Check if the error is caused by ANY of the missing optional packages
    error_msg = str(e).lower()
    known_extras = ['tensorly']
    
    if any(pkg in error_msg for pkg in known_extras):
        get_orbital = _missing_ext_stub
        get_basis = _missing_ext_stub
        tabulate = _missing_ext_stub
        rotation_matrix_from_euler = _missing_ext_stub
    else:
        # If the import failed for a completely different reason (e.g., a typo in .ext), raise it
        raise e

__all__ = ['reduce','image']