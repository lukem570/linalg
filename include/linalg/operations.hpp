/**
 * @file operations.hpp
 * @author lukem
 * @date 2025-11-28
 * @brief Macro defintions for TensorT's operations
 * 
 * Defintions for TensorT's operations as Macros
 */

#ifndef LINALG_OPERATIONS_HPP
#define LINALG_OPERATIONS_HPP

#define TENSOR_OPERATION(op, spec) \
    spec TensorT operator op(const TensorT& rhs) spec {         \
        TensorT newTensor;                                      \
                                                                \
        for (std::size_t i = 0; i < data.size(); i++) {         \
            newTensor[i] = (operator[](i) op rhs[i]);           \
        }                                                       \
                                                                \
        return newTensor;                                       \
    }

#define FLOAT_OPERATION(op, spec) \
    spec TensorT operator op(const float& rhs) spec {           \
        TensorT newTensor;                                      \
                                                                \
        for (std::size_t i = 0; i < data.size(); i++) {         \
            newTensor[i] = (operator[](i) op rhs);              \
        }                                                       \
                                                                \
        return newTensor;                                       \
    }

#define FLIPPED_FLOAT_VECTOR_OPERATION(op)                                               \
template <int D1>                                                                        \
TensorT<NumList<D1>> operator op(const float& lhs, const TensorT<NumList<D1>>& rhs) {    \
    TensorT<NumList<D1>> result;                                             \
    for (std::size_t i = 0; i < rhs.size(); ++i) {                                       \
        result[i] = lhs op rhs[i];                                                       \
    }                                                                                    \
    return result;                                                                       \
}

#define FLIPPED_FLOAT_TENSOR_OPERATION(op)                                                \
template <int... D>                                                                       \
TensorT<NumList<D...>> operator op(const float& lhs, const TensorT<NumList<D...>>& rhs) { \
    TensorT<NumList<D...>> result;                                            \
    for (std::size_t i = 0; i < rhs.size(); ++i) {                                        \
        result[i] = lhs op rhs[i];                                                        \
    }                                                                                     \
    return result;                                                                        \
}

#endif