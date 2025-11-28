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

    template <int D>
    class TensorT<NumList<D>> {
        public:

            TensorT() = default;
            TensorT(std::initializer_list<float> list);
            TensorT(float value);

            float& operator[](std::size_t index);
            const float& operator[](std::size_t index) const;
            float& getList(std::array<std::size_t, GetSize<NumList<D>>::value> indices);
            
            float dot(const TensorT& b) const;
            Vector<D> cross(Vector<D> other) const;
            
            float sum() const;
            float squaredLength() const;
            float length() const;

            Vector<D> normalize() const;
            
            Vector<D> lerp(Vector<D> to, float t) const;

            std::size_t size() const;
            std::string string() const;
            Vector<D+1> extend(float value) const;

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

        protected:
            std::array<float, D> data;
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