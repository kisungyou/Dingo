# Basic Operations

## Construction and Fill

```cpp
dingo::mat A(2, 3, dingo::fill::ones);
auto Z = dingo::zeros(2, 3);
auto R = dingo::randn(2, 3);
auto I = dingo::eye(3);
```

## Indexing and Slicing

```cpp
dingo::mat A{{1,2,3},{4,5,6},{7,8,9}};
auto sub = A.submat(dingo::range(1,2), dingo::range(0,1));
A(dingo::all_idx, dingo::range(0,0)) = 0.0;
```

Mask indexing:

```cpp
dingo::mat M{{1,2,3},{4,5,6}};
dingo::mat mask{{0,1,0},{1,0,1}};
auto picked = M(mask);
M(mask) = -1.0;
```

## Arithmetic

```cpp
dingo::mat A{{1,2},{3,4}};
dingo::mat B{{5,6},{7,8}};
auto sum = A + B;
auto prod = A * B; // matrix multiply
auto hadamard = A % B; // elementwise multiply
```

## Broadcasting

```cpp
dingo::mat M{{1,2,3},{4,5,6}};
dingo::vec c{10,20};
dingo::rowvec r{100,200,300};

auto a = M + c; // add by row
auto b = M + r; // add by column
```

## Reductions

```cpp
double total = dingo::sum(M);
auto col_sum = dingo::sum(M, 0);
auto row_sum = dingo::sum(M, 1);
```

Also available: `mean`, `min`, `max`, `var`, `stddev`, `trace`, `norm`, `any`, `all`.

## Reshape and Join

```cpp
auto v = dingo::vectorise(M);
auto reshaped = dingo::reshape(v.as_mat(), 3, 2);
auto hcat = dingo::join_rows(M, M);
auto vcat = dingo::join_cols(M, M);
```

## Structure Helpers

- `diagvec`, `diagmat`
- `tril`, `triu`
- `flipud`, `fliplr`
- `repmat`
- `trans`, `strans`

## Transpose

```cpp
auto t1 = dingo::trans(M);
auto t2 = dingo::strans(M);
```

`strans` is conjugate transpose for complex types.
