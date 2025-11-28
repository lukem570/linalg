/**
 * @file vector.hpp
 * @author lukem
 * @date 2025-11-28
 * @brief Base case for the TensorT class
 * 
 * This file contains the defintion for the base case of
 * the TensorT class, which happens to be a vector.
 */

#ifndef LINALG_VECTOR_HPP
#define LINALG_VECTOR_HPP

#include <array>
#include <initializer_list>
#include <iomanip>
#include <string>
#include <cmath>
#include <sstream>

#include <linalg/tensor.hpp>

namespace Linalg {

    template <int N>
    using Vector = Tensor<N>;

    template <int D1>
    class TensorT<NumList<D1>> {
        public:
            typedef NumList<D1> Dim;

            TensorT() = default;
            TensorT(std::initializer_list<float> list) {
                std::copy(list.begin(), list.end(), data.begin());
            }
            TensorT(float value) { data.fill(value); }

            float& operator[](std::size_t index) {
                return data[index];
            }

            const float& operator[](std::size_t index) const {
                return data[index];
            }

            TENSOR_OPERATION(+, const);
            TENSOR_OPERATION(-, const);
            TENSOR_OPERATION(*, const);
            TENSOR_OPERATION(/, const);

            FLOAT_OPERATION(+, const);
            FLOAT_OPERATION(-, const);
            FLOAT_OPERATION(*, const);
            FLOAT_OPERATION(/, const);

            TENSOR_OPERATION(+=,);
            TENSOR_OPERATION(-=,);
            TENSOR_OPERATION(*=,);
            TENSOR_OPERATION(/=,);

            FLOAT_OPERATION(+=,);
            FLOAT_OPERATION(-=,);
            FLOAT_OPERATION(*=,);
            FLOAT_OPERATION(/=,);

            float dot(const TensorT& b) const {
                float sum = 0.0f;
                for (int i = 0; i < D1; ++i)
                    sum += this->operator[](i)* b[i];
                return sum;
            }

            float sum() const {
                float sum = 0.0f;
                for (int i = 0; i < D1; ++i)
                    sum += this->operator[](i);
                return sum;
            }

            float squaredLength() const {
                return dot(*this);
            }

            float length() const {
                return std::sqrt(squaredLength());
            }

            float& getList(std::array<std::size_t, GetSize<Dim>::value> indices) {
                return data[indices.back()];
            }

            std::size_t size() const {
                return D1;
            }

            std::string string() const {
                std::stringstream stream;
                stream << "(";

                for (std::size_t i = 0; i < data.size() - 1; i++) {
                    stream << std::fixed << std::setprecision(6) << data[i] << ", ";
                }

                stream << std::fixed << std::setprecision(6) << data.back() << ")";

                return stream.str();
            }

            TensorT<NumList<D1>> lerp(TensorT<NumList<D1>> to, float t) {
                return *this * (1 - t) + to * t;
            }

            float determinant(const TensorT& other) const {
                static_assert(D1 == 2, "determinate only implemented for size 2");

                return data[0] * other[1] - data[1] * other[0];
            }

            TensorT<NumList<D1>> cross(TensorT<NumList<D1>> other) {
                static_assert(D1 == 3, "cross product only exits for a vector 3");

                return {
                    data[1] * other.data[2] - data[2] * other.data[1],
                    data[2] * other.data[0] - data[0] * other.data[2],
                    data[0] * other.data[1] - data[1] * other.data[0],
                };
            }

            TensorT<NumList<D1>> normalize() const {
                return this->operator/(length());
            }

            TensorT<NumList<D1 + 1>> extend(float value) const {
                TensorT<NumList<D1 + 1>> extended;
                for (std::size_t i = 0; i < D1; ++i)
                    extended[i] = this->operator[](i);
                extended[D1] = value;
                return extended;
            }

        protected:
            std::array<float, D1> data;

    };

    FLIPPED_FLOAT_VECTOR_OPERATION(+);
    FLIPPED_FLOAT_VECTOR_OPERATION(-);
    FLIPPED_FLOAT_VECTOR_OPERATION(*);
    FLIPPED_FLOAT_VECTOR_OPERATION(/);

    class Vec2 : public Vector<2> {
        public:
            using Vector<2>::TensorT;
            Vec2(Vector<2> r) : Vector<2>(r) {}
            Vec2 operator=(const Vec2& rhs) { 
                x = rhs.x;
                y = rhs.y;
                return *this;
            }

            float& x = data[0];
            float& y = data[1];
    };

    class Vec3 : public Vector<3> {
        public:
            using Vector<3>::TensorT;
            Vec3(Vector<3> r) : Vector<3>(r) {}
            Vec3 operator=(const Vec3& rhs) { 
                x = rhs.x;
                y = rhs.y;
                z = rhs.z;
                return *this;
            }

            float& x = data[0];
            float& y = data[1];
            float& z = data[2];
    };

    class Vec4 : public Vector<4> {
        public:
            using Vector<4>::TensorT;
            Vec4(Vector<4> r) : Vector<4>(r) {}
            Vec4 operator=(const Vec4& rhs) { 
                x = rhs.x;
                y = rhs.y;
                z = rhs.z;
                w = rhs.w;
                return *this;
            }

            float& x = data[0];
            float& y = data[1];
            float& z = data[2];
            float& w = data[3];
    };
}

#endif