/**
 * @file varargs.hpp
 * @author lukem
 * @date 2025-11-28
 * @brief Structures for managing variatic arguments
 * 
 * Uses template metaprogramming for managing variatic templates
 * Particularly in the TensorT class
 */

#ifndef LINALG_VARARGS_HPP
#define LINALG_VARARGS_HPP

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
}

#endif