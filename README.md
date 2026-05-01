# RV-Sparse: Sparse Matrix-Vector Multiplication in CSR Format

A C implementation of sparse matrix-vector multiplication using **Compressed Sparse Row (CSR)** format, with **zero dynamic memory allocation**.

---

## Problem Statement

Given a dense row-major matrix `A` and a vector `x`, implement a function that:

1. Scans `A` and identifies all non-zero elements
2. Extracts them into CSR format using caller-provided buffers
3. Computes the matrix-vector product `y = A * x` using the CSR representation
4. Writes the result into a caller-provided output buffer

**Critical constraint:** The function performs **zero dynamic memory allocation**. All buffers are pre-allocated by the caller.

---

## What is CSR Format?

Compressed Sparse Row (CSR) is a standard format for storing sparse matrices efficiently. Instead of storing all `rows × cols` elements, it stores only the non-zeros using three arrays:

| Array | Length | Description |
|---|---|---|
| `values` | `nnz` | Non-zero values, row by row |
| `col_indices` | `nnz` | Column index of each non-zero |
| `row_ptrs` | `rows + 1` | `row_ptrs[i]` = start index of row `i` in `values` |

For row `i`, its non-zeros live at `values[row_ptrs[i] .. row_ptrs[i+1] - 1]`.

**Example** - a 3×4 matrix:

```
A = [ 5  0  0  2 ]
    [ 0  0  3  0 ]
    [ 1  0  0  8 ]
```

```
values      = [ 5, 2, 3, 1, 8 ]
col_indices = [ 0, 3, 2, 0, 3 ]
row_ptrs    = [ 0, 2, 3, 5 ]
```

---

## Implementation

```c
void sparse_multiply(
    int rows, int cols, const double* A, const double* x,
    int* out_nnz, double* values, int* col_indices, int* row_ptrs,
    double* y
) {
    // --- Phase 1: Build CSR from dense matrix ---
    int nnz = 0;
    for (int i = 0; i < rows; ++i) {
        row_ptrs[i] = nnz;
        for (int j = 0; j < cols; ++j) {
            double val = A[i * cols + j];
            if (val != 0.0) {
                values[nnz] = val;
                col_indices[nnz] = j;
                nnz++;
            }
        }
    }
    row_ptrs[rows] = nnz;
    *out_nnz = nnz;

    // --- Phase 2: Compute y = A * x using CSR ---
    for (int i = 0; i < rows; ++i) {
        double dot = 0.0;
        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; ++k)
            dot += values[k] * x[col_indices[k]];
        y[i] = dot;
    }
}
```

### How it works

**Phase 1 - CSR construction:**
- Walk each row `i`; record `row_ptrs[i] = nnz` before scanning
- For every non-zero element found, store its value and column index, then increment `nnz`
- After all rows, set `row_ptrs[rows] = nnz` to close the last row's range
- Write the total non-zero count to `*out_nnz`

**Phase 2 - Matrix-vector product:**
- For each row `i`, iterate over `k` from `row_ptrs[i]` to `row_ptrs[i+1]`
- Accumulate `values[k] * x[col_indices[k]]` into a dot product
- Write the result to `y[i]`

---

## Build & Run

```bash
gcc -o run challenge.c -lm
./run
```

The test harness runs **100 randomised iterations**, each with a random matrix size (5–45 rows/cols) and density (5%–40%), verifying correctness against a naive dense reference computation.

---

## Test Results

All 100 iterations pass with **zero numerical error**, verified against a mixed absolute/relative tolerance of `1e-7 + 1e-7 * |y_ref[i]|`.

---

## Complexity

| | Dense | CSR (this implementation) |
|---|---|---|
| **Storage** | O(rows × cols) | O(nnz) |
| **Matvec** | O(rows × cols) | O(nnz) |

For sparse matrices where `nnz << rows × cols`, CSR provides significant memory and compute savings.

---

## File Structure

```
.
├── challenge.c   # Full implementation + test harness
└── README.md
```
