#!/usr/bin/env python3
import random

def generate_sparse_matrix(rows, cols, nnz_limit):
    """
    Generate a sparse matrix in row-wise CSR-like format.
    Each row is a list: [k, col1, val1, col2, val2, ...].
    """
    matrix = [[] for _ in range(rows)]
    nnz_total = 0

    while nnz_total < nnz_limit:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        v = random.randint(1, 9)  # small positive integers

        if not any(matrix[r][i] == c for i in range(1, len(matrix[r]), 2)):
            # add new non-zero element in row r
            matrix[r].append(c)
            matrix[r].append(v)
            nnz_total += 1

    # format: k col1 val1 col2 val2 ...
    formatted = []
    for row in matrix:
        k = len(row) // 2
        if k == 0:
            formatted.append("0")
        else:
            parts = [str(k)] + [str(x) for x in row]
            formatted.append(" ".join(parts))

    return formatted


def multiply_sparse(N, M, P, A, B):
    """
    Compute C = A * B in sparse row format.
    A: N x M
    B: M x P
    Both are given in row sparse format strings like ["k col val ..."].
    """
    # Parse into dict-of-dicts for quick multiplication
    A_parsed = []
    for row in A:
        items = row.split()
        if items[0] == "0":
            A_parsed.append({})
        else:
            d = {}
            for j in range(int(items[0])):
                col = int(items[2*j+1])
                val = int(items[2*j+2])
                d[col] = val
            A_parsed.append(d)

    B_parsed = []
    for row in B:
        items = row.split()
        if items[0] == "0":
            B_parsed.append({})
        else:
            d = {}
            for j in range(int(items[0])):
                col = int(items[2*j+1])
                val = int(items[2*j+2])
                d[col] = val
            B_parsed.append(d)

    # Multiply
    C_out = []
    for i in range(N):
        row_result = {}
        for a_col, a_val in A_parsed[i].items():
            for b_col, b_val in B_parsed[a_col].items():
                row_result[b_col] = row_result.get(b_col, 0) + a_val * b_val
        if not row_result:
            C_out.append("0")
        else:
            parts = [str(len(row_result))]
            for col, val in sorted(row_result.items()):
                parts.append(str(col))
                parts.append(str(val))
            C_out.append(" ".join(parts))
    return C_out


def main():
    N = M = P = 10001
    nnz_limit = 100000  # 1e5 non-zeros

    with open("sparse_testcase_large.txt", "w") as f:
        f.write("1\n")  # number of test cases
        f.write("# --- Test Case 1 ---\n")
        f.write("Input:\n")

        # Generate matrices
        A = generate_sparse_matrix(N, M, nnz_limit)
        B = generate_sparse_matrix(M, P, nnz_limit)

        f.write(f"{N} {M} {P}\n")
        for row in A:
            f.write(row + "\n")
        for row in B:
            f.write(row + "\n")

        f.write("Output:\n")
        C = multiply_sparse(N, M, P, A, B)
        for row in C:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
