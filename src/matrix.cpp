/**
 * @file matrix.cpp
 * @author lukem
 * @date 2025-11-28
 * @brief Implementation for Matrix functions
 */

#include <linalg/matrix.hpp>

namespace Linalg {

    template <int N, int V>
    Matrix<V, N> Matrix<N, V>::transpose() const {
        return this->template __permute<0, 1>();
    }

    template <int N, int V>
    float Matrix<N, V>::determinant() const {
        static_assert(N == V, "Determinant only defined for square matrices");

        if constexpr (N == 1) {
            return this->data[0][0];
        }

        float det = 0;
        int sign = 1;

        for (int f = 0; f < N; f++) {
            Matrix<N-1, N-1> temp;
            for (int i = 1; i < N; i++) {
                int j2 = 0;
                for (int j = 0; j < N; j++) {
                    if (j == f) continue;
                    temp.data[i-1][j2++] = this->data[i][j];
                }
            }
            det += sign * this->data[0][f] * temp.determinant();
            sign = -sign;
        }

        return det;
    }

    template <int N, int V>
    Matrix<N, V> Matrix<N, V>::adjoint() const {
        static_assert(N == V, "Adjoint only defined for square matrices");

        Matrix<N, V> adj;

        if constexpr (N == 1) {
            adj.data[0][0] = 1;
            return adj;
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                Matrix<N-1, N-1> temp;
                int rowIndex = 0;
                for (int row = 0; row < N; row++) {
                    if (row == i) continue;
                    int colIndex = 0;
                    for (int col = 0; col < N; col++) {
                        if (col == j) continue;
                        temp.data[rowIndex][colIndex++] = this->data[row][col];
                    }
                    rowIndex++;
                }

                int sign = ((i + j) % 2 == 0) ? 1 : -1;
                adj.data[j][i] = sign * temp.determinant();
            }
        }

        return adj;
    }

    template <int N, int V>
    Matrix<N, V> Matrix<N, V>::inverse() const {
        static_assert(N == V, "Inverse only defined for square matrices");

        float det = determinant();
        if (det == 0) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }

        Matrix<N, V> adj = adjoint();
        Matrix<N, V> inv;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                inv.data[i][j] = adj.data[i][j] / det;
            }
        }

        return inv;
    }

    template <int N, int V>
    Matrix<N, V> Matrix<N, V>::identity() {
        static_assert(N == V, "Identity only defined for square matrices");

        Matrix m = 0;

        for (std::size_t i = 0; i < N; i++) {
            m[i][i] = 1;
        }

        return m;
    }

    template <int N, int V>
    Vector<N> Matrix<N, V>::operator*(const Vector<V>& rhs) const {
        Vector<N> result;

        for (int i = 0; i < N; i++) {
            result[i] = 0;
            for (int j = 0; j < V; j++) {
                result[i] += this->data[i][j] * rhs[j];
            }
        }

        return result;
    }
}