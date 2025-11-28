/**
 * @file tensor.cpp
 * @author lukem
 * @date 2025-11-28
 * @brief Implementations for the TensorT main logic functions
 */

#include <linalg/tensor.hpp>

namespace Linalg {

    template <int ...D>
    Tensor<D...>::TensorT(std::initializer_list<TensorT<typename PopBack<NumList<D...>>::value>> list) {
        std::copy(list.begin(), list.end(), data.begin());
    }

    template <int ...D>
    Tensor<D...>::TensorT(float value) {
        data.fill(value);
    }

    template <std::size_t N>
    bool increment(std::array<std::size_t, N>& indices, const std::array<std::size_t, N>& shape) {
        for (int i = N - 1; i >= 0; --i) {
            if (++indices[i] < shape[i])
                return true;
            indices[i] = 0;
        }
        return false;
    }

    template <int D1, int D2, int ...D>
    TensorT<typename SwapItems<NumList<D...>, D1, D2>::value> __permuteFunc(TensorT<NumList<D...>> tensor) { 

        static_assert(D1 != D2, "Dimentions should not be the same.");

        TensorT<typename SwapItems<NumList<D...>, D1, D2>::value> newTensor;
        std::array<std::size_t, GetSize<NumList<D...>>::value> indices;
        std::array<std::size_t, GetSize<NumList<D...>>::value> shape_ = tensor.shape();
        indices.fill(0);

        do {

            std::size_t j = indices.size() - 1;
            while (indices[j] >= shape_[j]) {
                indices[j] = 0;
                j -= 1;
            }

            std::array<std::size_t, GetSize<NumList<D...>>::value> swappedIndices;
            swappedIndices = indices;
            std::swap(swappedIndices[D1], swappedIndices[D2]);

            newTensor.getList(swappedIndices) = tensor.getList(indices);

            indices.back() += 1;
        } while (increment(indices, shape_));

        return newTensor; 
    }

    template <int ...D>
    std::size_t Tensor<D...>::size() const {
        return GetItem<NumList<D...>, GetSize<NumList<D...>>::value-1>::element;
    }

    template <int ...D>
    std::array<std::size_t, GetSize<NumList<D...>>::value> Tensor<D...>::shape() const {
        return {D...};
    }

    template <int ...D>
    TensorT<typename PopBack<NumList<D...>>::value>& Tensor<D...>::operator[](std::size_t index) {
        return data[index];
    }

    template <int ...D>
    const TensorT<typename PopBack<NumList<D...>>::value>& Tensor<D...>::operator[](std::size_t index) const {
        return data[index];
    }

    template <int ...D>
    float& Tensor<D...>::getList(std::array<std::size_t, GetSize<NumList<D...>>::value> indices) {
        std::array<std::size_t, GetSize<NumList<D...>>::value-1> newIndices;

        for (std::size_t i = 0; i < newIndices.size(); i++) 
            newIndices[i] = indices[i];

        return data[indices.back()].getList(newIndices);
    }
}