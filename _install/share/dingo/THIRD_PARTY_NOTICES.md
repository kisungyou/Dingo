# Third-Party Notices

## Eigen (vendored)

- Project: Eigen
- Upstream: https://gitlab.com/libeigen/eigen
- Vendored location: `third_party/eigen`

Primary Eigen license is MPL-2.0. Some files in Eigen distributions may use additional
MPL-compatible licenses.

Bundled license files are provided in:

- `third_party/eigen/licenses/COPYING.README`
- `third_party/eigen/licenses/COPYING.MPL2`
- `third_party/eigen/licenses/COPYING.BSD`
- `third_party/eigen/licenses/COPYING.APACHE`
- `third_party/eigen/licenses/COPYING.MINPACK`

## BLAS/LAPACK in vendored Eigen tree

This repository vendors `third_party/eigen/blas` and `third_party/eigen/lapack` from the Eigen source tree.
These are treated as third-party code distributed with Eigen and covered by the bundled notice/license files above.

## Redistribution checklist

1. Keep original file headers and copyright/license notices in vendored third-party code.
2. Distribute the third-party license files listed above with source and binary artifacts.
3. If you modify MPL-covered files (for example Eigen files), provide the modified source of those files under MPL-2.0 to recipients.
4. Dingo project code remains under the top-level MIT `LICENSE`.

## Build-time guard

Dingo enables `EIGEN_MPL2_ONLY` by default (`DINGO_EIGEN_MPL2_ONLY=ON`) so consumer builds avoid Eigen components that are not MPL-2.0.
