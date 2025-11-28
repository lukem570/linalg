/**
 * @file tensor.hpp
 * @author lukem
 * @date 2025-11-28
 * @brief TensorT class main logic
 * 
 * This file contains the main logic for the TensorT class
 * Since the class definition is recursive the base case is
 * a vector
 */

#ifndef LINALG_TENSOR_HPP
#define LINALG_TENSOR_HPP

#include <array>
#include <initializer_list>

#include <linalg/varargs.hpp>
#include <linalg/operations.hpp>

#define permute(a, b) __permute<a, b>()

namespace Linalg {

    template <typename>
    class TensorT;

    template <int... Dims>
    using Tensor = TensorT<NumList<Dims...>>;

    template <int D1, int D2, int ...D>
    TensorT<typename SwapItems<NumList<D...>, D1, D2>::value> __permuteFunc(TensorT<NumList<D...>> tensor);

    template <int ...D>
    class TensorT<NumList<D...>> {
        public:

            TensorT() = default;
            TensorT(std::initializer_list<TensorT<typename PopBack<NumList<D...>>::value>> list);
            TensorT(float value);

            template <int D1, int D2>
            TensorT<typename SwapItems<NumList<D...>, D1, D2>::value> __permute() { return __permuteFunc<D1, D2, D...>(*this); }

            std::size_t size() const;
            std::array<std::size_t, GetSize<NumList<D...>>::value> shape() const;
            TensorT<typename PopBack<NumList<D...>>::value>& operator[](std::size_t index);
            const TensorT<typename PopBack<NumList<D...>>::value>& operator[](std::size_t index) const;
            float& getList(std::array<std::size_t, GetSize<NumList<D...>>::value> indices);
            
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
            std::array<
                TensorT<typename PopBack<NumList<D...>>::value>, 
                GetItem<NumList<D...>, GetSize<NumList<D...>>::value-1>::element
            > data;
    };

    FLIPPED_FLOAT_TENSOR_OPERATION(+);
    FLIPPED_FLOAT_TENSOR_OPERATION(-);
    FLIPPED_FLOAT_TENSOR_OPERATION(*);
    FLIPPED_FLOAT_TENSOR_OPERATION(/);
}

#endif