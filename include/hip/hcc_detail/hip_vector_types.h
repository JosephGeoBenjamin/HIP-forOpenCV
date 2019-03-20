/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*******************************************************************************
 * Modified Version - Vectors converted to Array for CV compatabilty
 *******************************************************************************/

#ifndef HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H
#define HIP_INCLUDE_HIP_HCC_DETAIL_HIP_VECTOR_TYPES_H

#if defined(__HCC__) && (__hcc_workweek__ < 16032)
#error("This version of HIP requires a newer version of HCC.");
#endif

#include "hip/hcc_detail/host_defines.h"

#if !defined(_MSC_VER)

//HIP_TODO:
//Disabled extended vectors due to OpenCV Issues
#if defined(__clang__)
    #define __NATIVE_VECTOR__(n, ...) __attribute__((ext_vector_type(n)))
#elif defined(__GNUC__) // N.B.: GCC does not support .xyzw syntax.
    #define __ROUND_UP_TO_NEXT_POT__(x) \
        (1 << (31 - __builtin_clz(x) + (x > (1 << (31 - __builtin_clz(x))))))
    #define __NATIVE_VECTOR__(n, T) \
        __attribute__((vector_size(__ROUND_UP_TO_NEXT_POT__(n) * sizeof(T))))
#endif

#if defined(__cplusplus)
    #include <type_traits>

// ====================== Modified

//------ Structure to mimic vector

    template <typename T, unsigned int n>
    struct data_holder_base{
        T dValue[n];
    };

    template <typename T>
    struct data_holder_base<T, 1>{
        union{
            T dValue[1];
            struct {
                T x;
            };
        };
    };

    template <typename T>
    struct data_holder_base<T, 2>{
        union{
            T dValue[2];
            struct {
                T x;
                T y;
            };
        };
    };

    template <typename T>
    struct data_holder_base<T, 3>{
        union{
            T dValue[3];
            struct {
                T x;
                T y;
                T z;
            };
        };
    };

    template <typename T>
    struct data_holder_base<T, 4>{
        union{
            T dValue[4];
            struct {
                T x;
                T y;
                T z;
                T w;
            };
        };
    };

    template<typename T, unsigned int rank>
    struct data_holder : public data_holder_base<T, rank> {

        using data_holder_base<T, rank>::dValue;

        //--- Constructors
        __host__ __device__
        data_holder() = default;

        template< typename U >
        __host__ __device__
        data_holder(U x) noexcept
        {
            for (auto i = 0u; i != rank; ++i) dValue[i] = x;
        };

        template< typename... Us >
        __host__ __device__
        data_holder(Us... xs) noexcept {
            T data_r[rank] = {static_cast<T>(xs)...};
            for(unsigned int i=0; i<rank; i++ )
                dValue[i] = data_r[i];
        };

        template< typename Ua >
        __host__ __device__
        data_holder(Ua* xa) noexcept {
            for(unsigned int i=0; i<rank; i++ )
                dValue[i] = xa[i];
        };

        __host__ __device__
        data_holder(const data_holder&) = default;
        __host__ __device__
        data_holder(data_holder&&) = default;

        __host__ __device__
        ~data_holder() = default;

        __host__ __device__
        data_holder& operator=(const data_holder&) = default;
        __host__ __device__
        data_holder& operator=(data_holder&&) = default;

        //--- Operator overload
        __host__ __device__
        T& operator[](const unsigned int index){
            return dValue[index];
        }
        __host__ __device__
        T operator[](const unsigned int index) const{
            return dValue[index];
        }
    };
//--------- Default structures

    template<typename T, unsigned int n>
    struct HIP_vector_base {
        typedef struct data_holder<T, n> Native_vec_;
        Native_vec_ data;
    };

    template<typename T>
    struct HIP_vector_base<T, 1> {
        typedef struct data_holder<T, 1> Native_vec_;

        union {
             Native_vec_ data;
            struct {
                T x;
            };
        };
    };

    template<typename T>
    struct HIP_vector_base<T, 2> {
        typedef struct data_holder<T, 2> Native_vec_ ;

        union {
            Native_vec_ data;
            struct {
                T x;
                T y;
            };
        };
    };

    template<typename T>
    struct HIP_vector_base<T, 3> {
         typedef struct data_holder<T, 3> Native_vec_;

        union {
         Native_vec_ data;
            struct {
                T x;
                T y;
                T z;
            };
        };
    };

    template<typename T>
    struct HIP_vector_base<T, 4> {
         typedef struct data_holder<T, 4> Native_vec_;

        union {
             Native_vec_ data;
            struct {
                T x;
                T y;
                T z;
                T w;
            };
        };
    };

    template<typename T, unsigned int rank>
    struct HIP_vector_type : public HIP_vector_base<T, rank> {
        using HIP_vector_base<T, rank>::data;
        using typename HIP_vector_base<T, rank>::Native_vec_;

        __host__ __device__
        HIP_vector_type() = default;
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type(U x) noexcept
        {
            for (auto i = 0u; i != rank; ++i) data[i] = x;
        }
        template< // TODO: constrain based on type as well.
            typename... Us,
            typename std::enable_if<
                (rank > 1) && sizeof...(Us) == rank>::type* = nullptr>
        __host__ __device__
        HIP_vector_type(Us... xs) noexcept {
            T data_r[rank] = {static_cast<T>(xs)...};
            for(unsigned int i=0; i<rank; i++ )
                data[i] = data_r[i];
            }

        template< typename Ua >
        __host__ __device__
        HIP_vector_type(Ua* xa) noexcept {
            for(unsigned int i=0; i<rank; i++ )
                data[i] = xa[i];
        };

        __host__ __device__
        HIP_vector_type(const HIP_vector_type&) = default;
        __host__ __device__
        HIP_vector_type(HIP_vector_type&&) = default;
        __host__ __device__
        ~HIP_vector_type() = default;

        __host__ __device__
        HIP_vector_type& operator=(const HIP_vector_type&) = default;
        __host__ __device__
        HIP_vector_type& operator=(HIP_vector_type&&) = default;

        // Arithmetic Operators

        __host__ __device__
        HIP_vector_type& operator++() noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                this->data[i] += 1;
            return *this;
        }
        __host__ __device__
        HIP_vector_type operator++(int) noexcept
        {
            auto tmp(*this);
            for(unsigned int i=0; i<rank; i++ )
                this->data[i] += 1;
            return tmp;
        }
        __host__ __device__
        HIP_vector_type& operator--() noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                this->data[i] -= 1;
            return *this;
        }
        __host__ __device__
        HIP_vector_type operator--(int) noexcept
        {
            auto tmp(*this);
            for(unsigned int i=0; i<rank; i++ )
                this->data[i] -= 1;
            return tmp;
        }
        __host__ __device__
        HIP_vector_type& operator+=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] += x.data[i];
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator+=(U x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] += x;
            return *this;
        }

        __host__ __device__
        HIP_vector_type& operator-=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] -= x.data[i];
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator-=(U x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] -= x;
            return *this;
        }
        __host__ __device__
        HIP_vector_type& operator*=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] *= x.data[i];
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator*=(U x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] *= x;
            return *this;
        }
        __host__ __device__
        HIP_vector_type& operator/=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] /= x.data[i];
            return *this;
        }
        template<
            typename U,
            typename std::enable_if<
                std::is_convertible<U, T>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator/=(U x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] /= x;
            return *this;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_signed<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type operator-() noexcept
        {
            auto tmp(*this);
            for(unsigned int i=0; i<rank; i++ )
                tmp.data[i] = -tmp.data[i];
            return tmp;
        }

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type operator~() noexcept
        {
            T data_not[rank];
            for(unsigned int i=0; i<rank; i++ )
                data_not[i] = ~this->data[i];
            return data_not;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator%=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] %= x.data[i];
            return *this;
        }

        // Bitwise Operators

        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator^=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] ^= x.data[i];
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator|=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] |= x.data[i];
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator&=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] &= x.data[i];
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator>>=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] >>= x.data[i];
            return *this;
        }
        template<
            typename U = T,
            typename std::enable_if<std::is_integral<U>{}>::type* = nullptr>
        __host__ __device__
        HIP_vector_type& operator<<=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                data[i] <<= x.data[i];
            return *this;
        }

        __host__ __device__
        bool operator==(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                if (!(data[i] == x.data[i]))
                    return false;
            return true;
        }

        __host__ __device__
        bool operator!=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                if (!(data[i] != x.data[i]))
                    return false;
            return true;
        }

        __host__ __device__
        bool operator<=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                if (!(data[i] <= x.data[i]))
                    return false;
            return true;
        }

        __host__ __device__
        bool operator>=(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                if (!(data[i] >= x.data[i]))
                    return false;
            return true;
        }

        __host__ __device__
        bool operator<(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                if (!(data[i] < x.data[i]))
                    return false;
            return true;
        }

        __host__ __device__
        bool operator>(const HIP_vector_type& x) noexcept
        {
            for(unsigned int i=0; i<rank; i++ )
                if (!(data[i] > x.data[i]))
                    return false;
            return true;
        }

    }; //Actual Class end

//==============================================================================

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator+(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} += y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator+(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return y + x;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator-(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator-(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} -= y;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator*(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator*(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} *= y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator*(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return y * x;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator/(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator/(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    HIP_vector_type<T, n> operator/(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} /= y;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator==(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        auto tmp = x.data == y.data;
        for (auto i = 0u; i != n; ++i) if (tmp[i] == 0) return false;
        return true;
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    bool operator==(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return x == HIP_vector_type<T, n>{y};
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    bool operator==(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} == y;
    }

    template<typename T, unsigned int n>
    __host__ __device__
    inline
    bool operator!=(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    bool operator!=(const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return !(x == y);
    }
    template<typename T, unsigned int n, typename U>
    __host__ __device__
    inline
    bool operator!=(U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return !(x == y);
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator%(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator%(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} %= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator^(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator^(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} ^= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator|(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator|(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} |= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator&(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator&(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} &= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator>>(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator>>(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} >>= y;
    }

    template<
        typename T,
        unsigned int n,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator<<(
        const HIP_vector_type<T, n>& x, U y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }
    template<
        typename T,
        unsigned int n,
        typename U,
        typename std::enable_if<std::is_integral<T>{}>* = nullptr>
    inline
    HIP_vector_type<T, n> operator<<(
        U x, const HIP_vector_type<T, n>& y) noexcept
    {
        return HIP_vector_type<T, n>{x} <<= y;
    }

    #define __MAKE_VECTOR_TYPE__(CUDA_name, T) \
        using CUDA_name##1 = HIP_vector_type<T, 1>;\
        using CUDA_name##2 = HIP_vector_type<T, 2>;\
        using CUDA_name##3 = HIP_vector_type<T, 3>;\
        using CUDA_name##4 = HIP_vector_type<T, 4>;
#else
    #define __MAKE_VECTOR_TYPE__(CUDA_name, T) \
        typedef T CUDA_name##_impl1 __NATIVE_VECTOR__(1, T);\
        typedef T CUDA_name##_impl2 __NATIVE_VECTOR__(2, T);\
        typedef T CUDA_name##_impl3 __NATIVE_VECTOR__(3, T);\
        typedef T CUDA_name##_impl4 __NATIVE_VECTOR__(4, T);\
        typedef struct {\
            union {\
                CUDA_name##_impl1 data;\
                struct {\
                    T x;\
                };\
            };\
        } CUDA_name##1;\
        typedef struct {\
            union {\
                CUDA_name##_impl2 data;\
                struct {\
                    T x;\
                    T y;\
                };\
            };\
        } CUDA_name##2;\
        typedef struct {\
            union {\
                CUDA_name##_impl3 data;\
                struct {\
                    T x;\
                    T y;\
                    T z;\
                };\
            };\
        } CUDA_name##3;\
        typedef struct {\
            union {\
                CUDA_name##_impl4 data;\
                struct {\
                    T x;\
                    T y;\
                    T z;\
                    T w;\
                };\
            };\
        } CUDA_name##4;
#endif

__MAKE_VECTOR_TYPE__(uchar, unsigned char);
__MAKE_VECTOR_TYPE__(char, char);
__MAKE_VECTOR_TYPE__(ushort, unsigned short);
__MAKE_VECTOR_TYPE__(short, short);
__MAKE_VECTOR_TYPE__(uint, unsigned int);
__MAKE_VECTOR_TYPE__(int, int);
__MAKE_VECTOR_TYPE__(ulong, unsigned long);
__MAKE_VECTOR_TYPE__(long, long);
__MAKE_VECTOR_TYPE__(ulonglong, unsigned long long);
__MAKE_VECTOR_TYPE__(longlong, long long);
__MAKE_VECTOR_TYPE__(float, float);
__MAKE_VECTOR_TYPE__(double, double);

#define DECLOP_MAKE_ONE_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x) { type r = {x}; return r; }

#define DECLOP_MAKE_TWO_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y) { type r = {x, y}; return r; }

#define DECLOP_MAKE_THREE_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y, comp z) { type r = {x, y, z}; return r; }

#define DECLOP_MAKE_FOUR_COMPONENT(comp, type) \
    __device__ __host__ \
    static \
    inline \
    type make_##type(comp x, comp y, comp z, comp w) { \
        type r = {x, y, z, w}; \
        return r; \
    }

DECLOP_MAKE_ONE_COMPONENT(unsigned char, uchar1);
DECLOP_MAKE_TWO_COMPONENT(unsigned char, uchar2);
DECLOP_MAKE_THREE_COMPONENT(unsigned char, uchar3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned char, uchar4);

DECLOP_MAKE_ONE_COMPONENT(signed char, char1);
DECLOP_MAKE_TWO_COMPONENT(signed char, char2);
DECLOP_MAKE_THREE_COMPONENT(signed char, char3);
DECLOP_MAKE_FOUR_COMPONENT(signed char, char4);

DECLOP_MAKE_ONE_COMPONENT(unsigned short, ushort1);
DECLOP_MAKE_TWO_COMPONENT(unsigned short, ushort2);
DECLOP_MAKE_THREE_COMPONENT(unsigned short, ushort3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned short, ushort4);

DECLOP_MAKE_ONE_COMPONENT(signed short, short1);
DECLOP_MAKE_TWO_COMPONENT(signed short, short2);
DECLOP_MAKE_THREE_COMPONENT(signed short, short3);
DECLOP_MAKE_FOUR_COMPONENT(signed short, short4);

DECLOP_MAKE_ONE_COMPONENT(unsigned int, uint1);
DECLOP_MAKE_TWO_COMPONENT(unsigned int, uint2);
DECLOP_MAKE_THREE_COMPONENT(unsigned int, uint3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned int, uint4);

DECLOP_MAKE_ONE_COMPONENT(signed int, int1);
DECLOP_MAKE_TWO_COMPONENT(signed int, int2);
DECLOP_MAKE_THREE_COMPONENT(signed int, int3);
DECLOP_MAKE_FOUR_COMPONENT(signed int, int4);

DECLOP_MAKE_ONE_COMPONENT(float, float1);
DECLOP_MAKE_TWO_COMPONENT(float, float2);
DECLOP_MAKE_THREE_COMPONENT(float, float3);
DECLOP_MAKE_FOUR_COMPONENT(float, float4);

DECLOP_MAKE_ONE_COMPONENT(double, double1);
DECLOP_MAKE_TWO_COMPONENT(double, double2);
DECLOP_MAKE_THREE_COMPONENT(double, double3);
DECLOP_MAKE_FOUR_COMPONENT(double, double4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long, ulong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long, ulong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long, ulong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long, ulong4);

DECLOP_MAKE_ONE_COMPONENT(signed long, long1);
DECLOP_MAKE_TWO_COMPONENT(signed long, long2);
DECLOP_MAKE_THREE_COMPONENT(signed long, long3);
DECLOP_MAKE_FOUR_COMPONENT(signed long, long4);

DECLOP_MAKE_ONE_COMPONENT(unsigned long long, ulonglong1);
DECLOP_MAKE_TWO_COMPONENT(unsigned long long, ulonglong2);
DECLOP_MAKE_THREE_COMPONENT(unsigned long long, ulonglong3);
DECLOP_MAKE_FOUR_COMPONENT(unsigned long long, ulonglong4);

DECLOP_MAKE_ONE_COMPONENT(signed long long, longlong1);
DECLOP_MAKE_TWO_COMPONENT(signed long long, longlong2);
DECLOP_MAKE_THREE_COMPONENT(signed long long, longlong3);
DECLOP_MAKE_FOUR_COMPONENT(signed long long, longlong4);
#else // defined(_MSC_VER)
#include <mmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

typedef union { char data; } char1;
typedef union { char data[2]; } char2;
typedef union { char data[4]; } char4;
typedef union { char4 data; } char3;
typedef union { __m64 data; } char8;
typedef union { __m128i data; } char16;

typedef union { unsigned char data; } uchar1;
typedef union { unsigned char data[2]; } uchar2;
typedef union { unsigned char data[4]; } uchar4;
typedef union { uchar4 data; } uchar3;
typedef union { __m64 data; } uchar8;
typedef union { __m128i data; } uchar16;

typedef union { short data; } short1;
typedef union { short data[2]; } short2;
typedef union { __m64 data; } short4;
typedef union { short4 data; } short3;
typedef union { __m128i data; } short8;
typedef union { __m128i data[2]; } short16;

typedef union { unsigned short data; } ushort1;
typedef union { unsigned short data[2]; } ushort2;
typedef union { __m64 data; } ushort4;
typedef union { ushort4 data; } ushort3;
typedef union { __m128i data; } ushort8;
typedef union { __m128i data[2]; } ushort16;

typedef union { int data; } int1;
typedef union { __m64 data; } int2;
typedef union { __m128i data; } int4;
typedef union { int4 data; } int3;
typedef union { __m128i data[2]; } int8;
typedef union { __m128i data[4];} int16;

typedef union { unsigned int data; } uint1;
typedef union { __m64 data; } uint2;
typedef union { __m128i data; } uint4;
typedef union { uint4 data; } uint3;
typedef union { __m128i data[2]; } uint8;
typedef union { __m128i data[4]; } uint16;

#if !defined(_WIN64)
typedef union { int data; } long1;
typedef union { __m64 data; } long2;
typedef union { __m128i data; } long4;
typedef union { long4 data; } long3;
typedef union { __m128i data[2]; } long8;
typedef union { __m128i data[4]; } long16;

typedef union { unsigned int data; } ulong1;
typedef union { __m64 data; } ulong2;
typedef union { __m128i data; } ulong4;
typedef union { ulong4 data; } ulong3;
typedef union { __m128i data[2]; } ulong8;
typedef union { __m128i data[4]; } ulong16;
#else // defined(_WIN64)
typedef union { __m64 data; } long1;
typedef union { __m128i data; } long2;
typedef union { __m128i data[2]; } long4;
typedef union { long4 data; } long3;
typedef union { __m128i data[4]; } long8;
typedef union { __m128i data[8]; } long16;

typedef union { __m64 data; } ulong1;
typedef union { __m128i data; } ulong2;
typedef union { __m128i data[2]; } ulong4;
typedef union { ulong4 data; } ulong3;
typedef union { __m128i data[4]; } ulong8;
typedef union { __m128i data[8]; } ulong16;
#endif // defined(_WIN64)

typedef union { __m64 data; } longlong1;
typedef union { __m128i data; } longlong2;
typedef union { __m128i data[2]; } longlong4;
typedef union { longlong4 data; } longlong3;
typedef union { __m128i data[4]; } longlong8;
typedef union { __m128i data[8]; } longlong16;

typedef union { __m64 data; } ulonglong1;
typedef union { __m128i data; } ulonglong2;
typedef union { __m128i data[2]; } ulonglong4;
typedef union { ulonglong4 data; } ulonglong3;
typedef union { __m128i data[4]; } ulonglong8;
typedef union { __m128i data[8]; } ulonglong16;

typedef union { float data; } float1;
typedef union { __m64 data; } float2;
typedef union { __m128 data; } float4;
typedef union { float4 data; } float3;
typedef union { __m256 data; } float8;
typedef union { __m256 data[2]; } float16;

typedef union { double data; } double1;
typedef union { __m128d data; } double2;
typedef union { __m256d data; } double4;
typedef union { double4 data; } double3;
typedef union { __m256d data[2]; } double8;
typedef union { __m256d data[4]; } double16;

#endif // defined(_MSC_VER)
#endif