# Getting Started

## 1. Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## 2. Include Dingo

Use the simplified include style:

```cpp
#include <dingo>
```

## 3. First Program

```cpp
#include <dingo>
#include <iostream>

int main() {
  dingo::mat A{{4.0, 2.0}, {1.0, 3.0}};
  dingo::vec b{1.0, 2.0};

  dingo::vec x = dingo::solve(A, b);
  std::cout << "x = [" << x(0) << ", " << x(1) << "]\n";
  return 0;
}
```

## 4. Quick Indexing Example

```cpp
dingo::mat M{{1,2,3},{4,5,6},{7,8,9}};
auto block = M(dingo::range(0,1), dingo::range(1,2));
M(dingo::all_idx, dingo::range(0,0)) = 0.0;
```

## 5. Next Steps

- Core data types: [Core Types](core-types.md)
- Common operations: [Basic Operations](basic-operations.md)
- Solvers/decompositions: [Linear Algebra](linear-algebra.md)
- Full API: [API Reference](api-reference/index.md)
