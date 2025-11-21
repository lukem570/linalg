#ifndef LINALG_HPP
#define LINALG_HPP

#include <cmath>
#include <array>
#include <string>
#include <sstream>
#include <iomanip>

namespace Linalg {}
namespace la = Linalg;

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

#define permute(a, b) __permute<a, b>()

namespace Linalg {

    template<int...>
    struct NumList;

    template<int T, int... TT>
    struct NumList<T, TT...>  {
        enum {head = T};
        typedef NumList<TT...> tail;
    };

    template<>
    struct NumList<> {};

    template <typename...>
    struct ConcatList;

    template <int... First, int ...Second>
    struct ConcatList<NumList<First...>, NumList<Second...>> {
        typedef NumList<First..., Second...> value;
    };

    template <int, typename...>
    struct Prepend;

    template <int First, int ...Second>
    struct Prepend<First, NumList<Second...>> {
        typedef NumList<First, Second...> value;
    };

    template<typename List>
    struct GetSize;

    template <int... T>
    struct GetSize<NumList<T...>> {
        enum {value = sizeof...(T)};
    };

    template <typename List, int N, typename = void>
    struct GetItem {
        static_assert(N > 0, "index cannot be negative");
        static_assert(GetSize<List>::value > 0, "index too high");
        enum {element = GetItem<typename List::tail, N-1>::element};
    };

    template <typename List>
    struct GetItem<List, 0> {
        static_assert(GetSize<List>::value > 0, "index too high");
        enum {element = List::head};
    };

    template <typename List, int N, int V>
    struct SetItem {
        static_assert(N > 0, "index cannot be negative");
        static_assert(GetSize<List>::value > 0, "index too high");
        typedef typename Prepend<List::head, 
                                typename SetItem<typename List::tail, N-1, V>::value>
            ::value value;
    };

    template <int New, int Old, int ...T>
    struct SetItem<NumList<Old, T...>, 0, New> {
        typedef NumList<New, T...> value;
    };

    template<typename List, int A, int B>
    struct SwapItems {
        typedef typename SetItem<typename SetItem<List, A, GetItem<List, B>::element>::value,
                    B, GetItem<List, A>::element>::value value;
    };

    template<typename>
    struct PopBack;

    template <int... N>
    struct PopBack<NumList<N...>> {
        typedef typename Prepend<NumList<N...>::head, 
                            typename PopBack<typename NumList<N...>::tail>::value>::value value;
    };

    template<int T>
    struct PopBack<NumList<T>> {
        typedef NumList<> value;
    };

    template <typename>
    class TensorT;

    template <>
    class TensorT<NumList<>> {
        public:
            operator float() { return data; }

        private:
            float data;
    };

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

        private:
            std::array<float, D1> data;

    };

    template <int ...D>
    class TensorT<NumList<D...>> {
        public:
            typedef NumList<D...> Dim;

            TensorT() = default;
            TensorT(std::initializer_list<TensorT<typename PopBack<Dim>::value>> list) {
                std::copy(list.begin(), list.end(), data.begin());
            }
            TensorT(float value) { data.fill(value); }

            template <int D1, int D2>
            TensorT<typename SwapItems<Dim, D1, D2>::value> __permute() { 

                static_assert(D1 != D2, "Dimentions should not be the same.");

                TensorT<typename SwapItems<Dim, D1, D2>::value> newTensor;
                std::array<std::size_t, GetSize<Dim>::value> indices;
                std::array<std::size_t, GetSize<Dim>::value> shape_ = shape();
                indices.fill(0);

                do {

                    std::size_t j = indices.size() - 1;
                    while (indices[j] >= shape_[j]) {
                        indices[j] = 0;
                        j -= 1;
                    }

                    std::array<std::size_t, GetSize<Dim>::value> swappedIndices;
                    swappedIndices = indices;
                    std::swap(swappedIndices[D1], swappedIndices[D2]);

                    newTensor.getList(swappedIndices) = getList(indices);

                    indices.back() += 1;
                } while (increment(indices, shape_));

                return newTensor; 
            }

            std::size_t size() const {
                return GetItem<Dim, GetSize<Dim>::value-1>::element;
            }

            const std::array<std::size_t, GetSize<Dim>::value> shape() const {
                return {D...};
            }

            TensorT<typename PopBack<Dim>::value>& operator[](std::size_t index) {
                return data[index];
            }

            const TensorT<typename PopBack<Dim>::value>& operator[](std::size_t index) const {
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
            
            float& getList(std::array<std::size_t, GetSize<Dim>::value> indices) {
                std::array<std::size_t, GetSize<Dim>::value-1> newIndices;

                for (std::size_t i = 0; i < newIndices.size(); i++) 
                    newIndices[i] = indices[i];

                return data[indices.back()].getList(newIndices);
            }

        private:
            std::array<
                TensorT<typename PopBack<Dim>::value>, 
                GetItem<Dim, GetSize<Dim>::value-1>::element
            > data;

            template <std::size_t N>
            bool increment(std::array<std::size_t, N>& indices, const std::array<std::size_t, N>& shape) {
                for (int i = N - 1; i >= 0; --i) {
                    if (++indices[i] < shape[i])
                        return true;
                    indices[i] = 0;
                }
                return false;
            }
    };

    FLIPPED_FLOAT_VECTOR_OPERATION(+);
    FLIPPED_FLOAT_VECTOR_OPERATION(-);
    FLIPPED_FLOAT_VECTOR_OPERATION(*);
    FLIPPED_FLOAT_VECTOR_OPERATION(/);

    FLIPPED_FLOAT_TENSOR_OPERATION(+);
    FLIPPED_FLOAT_TENSOR_OPERATION(-);
    FLIPPED_FLOAT_TENSOR_OPERATION(*);
    FLIPPED_FLOAT_TENSOR_OPERATION(/);
    
    template <int... Dims>
    using Tensor = TensorT<NumList<Dims...>>;
    
    template <int N>
    using Vector = Tensor<N>;
    
    template <int M, int N>
    using Matrix = Tensor<M, N>;

    inline Vector<3> cross(const Vector<3>& a, const Vector<3>& b) {
        Vector<3> result;
        result[0] = a[1]*b[2] - a[2]*b[1];
        result[1] = a[2]*b[0] - a[0]*b[2];
        result[2] = a[0]*b[1] - a[1]*b[0];
        return result;
    }

    template <int M, int N>
    Matrix<N,M> transpose(const Matrix<M,N>& A) {
        return A.template permute(0, 1);
    }

    template <int M, int N>
    Vector<M> matvec(const Matrix<M,N>& A, const Vector<N>& x) {
        Vector<M> y;
        for (int i = 0; i < M; ++i) {
            y[i] = 0;
            for (int j = 0; j < N; ++j)
                y[i] += A[i][j] * x[j];
        }
        return y;
    }

    template <int M, int N, int P>
    Matrix<M,P> matmul(const Matrix<M,N>& A, const Matrix<N,P>& B) {
        Matrix<M,P> C;
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < P; ++j) {
                C[i][j] = 0;
                for (int k = 0; k < N; ++k)
                    C[i][j] += A[i][k] * B[k][j];
            }
        return C;
    }

    using Vec1 = Vector<1>;
    using Vec2 = Vector<2>;
    using Vec3 = Vector<3>;
    using Vec4 = Vector<4>;
}

#endif