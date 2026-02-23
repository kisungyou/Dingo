# Core Types

## Mental Model

| Type | Shape | Use case |
|---|---|---|
| `dingo::Mat<T>` | 2D matrix | General dense matrix work |
| `dingo::Col<T>` | Nx1 vector | Column vectors |
| `dingo::Row<T>` | 1xN vector | Row vectors |
| `dingo::Cube<T>` | 3D tensor-like container | Multi-slice data |
| `dingo::Cell<T>` | Array-like container | Store arbitrary `T` values |

Convenient aliases:

- `dingo::mat`, `dingo::vec`, `dingo::rowvec`
- `dingo::cx_mat`, `dingo::cx_vec`, `dingo::cx_rowvec`

## `Mat`, `Col`, `Row`

- `Mat` is the main dense type.
- `Col` and `Row` are specialized vector types.
- They support fills (`zeros`, `ones`, `randu`, `randn`) and indexing.

```cpp
dingo::mat A(3, 4, dingo::fill::zeros);
dingo::vec v{1.0, 2.0, 3.0};
dingo::rowvec r{10.0, 20.0, 30.0, 40.0};
```

## `Cube`

`Cube` stores multiple matrix slices.

```cpp
dingo::cube C(2, 3, 4, dingo::fill::zeros);
C(1, 2, 3) = 5.0;
auto S = C.slice(3); // S is Mat
```

## `Cell<T>`

`Cell<T>` stores elements of any type `T` in a 1D/2D/3D indexed layout.

```cpp
dingo::cell<std::string> names(2, 2);
names(0, 0) = "alice";
names(1, 1) = "bob";
```

## Common Confusion

- `Mat` linear indexing (`A(i)`) is column-major.
- `Cube` index order is `(row, col, slice)`.
- `Cell<T>` is not numeric-only; it can hold any type.
