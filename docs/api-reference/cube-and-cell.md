# Cube and Cell API

## `Cube<T>`

### Constructors

- **What it does:** Create 3D dense container by `(rows, cols, slices)` and optional fill mode.
- **Inputs:** Dimensions and optional `fill::zeros/ones/randu/randn`.
- **Returns:** `Cube<T>`.
- **Simple example:**
  ```cpp
  dingo::cube C(2, 3, 4, dingo::fill::zeros);
  ```
- **Common mistake:** Confusing slice count with column count.

### Size and access

APIs: `n_rows()`, `n_cols()`, `n_slices()`, `n_elem()`, `resize(rows, cols, slices)`, `operator()(row,col,slice)`, `slice(i)`.

- **What it does:** Query/resize/access cube data.
- **Inputs:** Indices and new dimensions.
- **Returns:** Sizes, references, or slice matrices.
- **Simple example:**
  ```cpp
  C(1, 2, 3) = 7.0;
  auto S = C.slice(3);
  ```
- **Common mistake:** Using invalid slice index.

### Fill mutators

APIs: `zeros()`, `ones()`, `randu()`, `randn()`.

- **What it does:** Fill all slices in-place.
- **Inputs:** None.
- **Returns:** `void`.
- **Simple example:**
  ```cpp
  C.ones();
  ```
- **Common mistake:** Assuming random fill is deterministic without seed control.

## Cube Helper Functions

### `squeeze(cube)`

- **What it does:** Converts `Cube<T>` with exactly one slice into `Mat<T>`.
- **Inputs:** Cube with `n_slices()==1`.
- **Returns:** `Mat<T>`.
- **Simple example:**
  ```cpp
  dingo::cube C(2,3,1, dingo::fill::zeros);
  auto M = dingo::squeeze(C);
  ```
- **Common mistake:** Calling on multi-slice cube.

### `permute(cube, d1, d2, d3)`

- **What it does:** Reorders cube dimensions (1-based dimension ids).
- **Inputs:** Cube, three unique dims in `{1,2,3}`.
- **Returns:** New `Cube<T>`.
- **Simple example:**
  ```cpp
  auto P = dingo::permute(C, 3, 2, 1);
  ```
- **Common mistake:** Repeating dimensions or using 0-based dim ids.

### `join_slices(a, b)`

- **What it does:** Concatenates cubes along slice dimension.
- **Inputs:** Cubes with matching `n_rows` and `n_cols`.
- **Returns:** Combined `Cube<T>`.
- **Simple example:**
  ```cpp
  auto J = dingo::join_slices(A, B);
  ```
- **Common mistake:** Trying to join cubes with different matrix shape.

## `Cell<T>`

### Constructors and size

APIs: default, `Cell(n_elem)`, `Cell(rows, cols)`, `Cell(rows, cols, slices)`, plus `n_rows`, `n_cols`, `n_slices`, `n_elem`, `is_empty`, `resize`.

- **What it does:** Generic container with 1D/2D/3D indexing.
- **Inputs:** Dimensions.
- **Returns:** Cell container and shape info.
- **Simple example:**
  ```cpp
  dingo::cell<std::string> names(2,2);
  ```
- **Common mistake:** Forgetting `resize` resets storage layout.

### Element access and fill

APIs: `operator()(idx)`, `operator()(row,col,slice=0)`, `fill(value)`.

- **What it does:** Read/write elements and bulk-fill values.
- **Inputs:** Linear or multi-index; fill value.
- **Returns:** Element references or `void`.
- **Simple example:**
  ```cpp
  names(0,0) = "alice";
  names.fill("unknown");
  ```
- **Common mistake:** Out-of-range access after shape changes.
