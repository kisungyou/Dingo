# Limitations

This reflects current v1 scope.

## Not Included Yet

- File I/O APIs
- Sparse matrix APIs
- `kmeans`
- Gaussian mixture model estimation

## Additional Current Constraints

- `norm` currently supports only `p = 0, 1, 2`.
- `squeeze` currently requires `n_slices == 1`.
- `lu` currently requires a square input matrix.
- Some decomposition APIs return `bool`; callers should check success.
