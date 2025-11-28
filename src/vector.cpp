/**
 * @file vector.cpp
 * @author lukem
 * @date 2025-11-28
 * @brief Implementations for the vector functions
 */

#include <linalg/vector.hpp>

namespace Linalg {

    template <int D>
    Vector<D>::TensorT(std::initializer_list<float> list) {
        std::copy(list.begin(), list.end(), data.begin());
    }

    template <int D>
    Vector<D>::TensorT(float value) {
        data.fill(value);
    }

    template <int D>
    float& Vector<D>::operator[](std::size_t index) {
        return data[index];
    }

    template <int D>
    const float& Vector<D>::operator[](std::size_t index) const {
        return data[index];
    }

    template <int D>
    float Vector<D>::dot(const TensorT& b) const {
        float sum = 0.0f;
        for (int i = 0; i < D; ++i)
        sum += this->operator[](i)* b[i];
        return sum;
    }
    
    template <int D>
    float Vector<D>::sum() const {
        float sum = 0.0f;
        for (int i = 0; i < D; ++i)
        sum += this->operator[](i);
        return sum;
    }
    
    template <int D>
    float Vector<D>::squaredLength() const {
        return dot(*this);
    }
    
    template <int D>
    float Vector<D>::length() const {
        return std::sqrt(squaredLength());
    }
    
    template <int D>
    float& Vector<D>::getList(std::array<std::size_t, GetSize<NumList<D>>::value> indices) {
        return data[indices.back()];
    }
    
    template <int D>
    std::size_t Vector<D>::size() const {
        return D;
    }
    
    template <int D>
    std::string Vector<D>::string() const {
        std::stringstream stream;
        stream << "(";
        
        for (std::size_t i = 0; i < data.size() - 1; i++) {
            stream << std::fixed << std::setprecision(6) << data[i] << ", ";
        }
        
        stream << std::fixed << std::setprecision(6) << data.back() << ")";
        
        return stream.str();
    }
    
    template <int D>
    Vector<D> Vector<D>::lerp(Vector<D> to, float t) const {
        return *this * (1 - t) + to * t;
    }
    
    template <int D>
    Vector<D> Vector<D>::cross(Vector<D> other) const {
        static_assert(D == 3, "cross product only exits for a vector 3");
        
        return {
            data[1] * other.data[2] - data[2] * other.data[1],
            data[2] * other.data[0] - data[0] * other.data[2],
            data[0] * other.data[1] - data[1] * other.data[0],
        };
    }
    
    template <int D>
    Vector<D> Vector<D>::normalize() const {
        return this->operator/(length());
    }
    
    template <int D>
    Vector<D+1> Vector<D>::extend(float value) const {
        Vector<D+1> extended;
        for (std::size_t i = 0; i < D; ++i)
        extended[i] = this->operator[](i);
        extended[D] = value;
        return extended;
    }
}