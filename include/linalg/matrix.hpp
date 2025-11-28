/**
 * @file matrix.hpp
 * @author lukem
 * @date 2025-11-28
 * @brief Inherited class of Tensor with two dimentions
 * 
 * Contains the Matrix class which is a child of the 
 * Tensor class. 
 */

#ifndef LINALG_MATRIX_HPP
#define LINALG_MATRIX_HPP

#include <linalg/tensor.hpp>
#include <linalg/vector.hpp>

namespace Linalg {

    template <int N, int V> 
    class Matrix : public Tensor<N, V> {
        public:
            using Tensor<N, V>::TensorT;
            Matrix(Tensor<N, V> r) : Tensor<N, V>(r) {}
            Matrix operator=(const Matrix& rhs) { 
                this->data = rhs.data;
                return *this;
            }

            Matrix<V, N> transpose() const;
            Matrix adjoint() const;
            Matrix inverse() const;
            float determinant() const;

            Vector<N> operator*(const Vector<V>& rhs) const;

            static Matrix identity();
    };

    using Mat2 = Matrix<2, 2>;
    using Mat3 = Matrix<3, 3>;
    using Mat4 = Matrix<4, 4>;
}

#endif