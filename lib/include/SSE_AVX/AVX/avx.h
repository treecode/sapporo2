#pragma once

#ifndef __AVX__
#error "Please enable AVX support in g++"
#endif

#define __out
#include <cassert>

/* definitions */
namespace SIMD
{
  template<class T>
    struct vector;

  template<class T>
    struct scalar
    {
      const T &v;
      scalar() {}
      scalar(const T &_v) : v(_v) {}
      operator const T&() const {return v;}

      static inline const T& aload(const T &x) {return x;};
      static inline const T& uload(const T &x) {return x;};
      static inline T& aref(T &x) {return x;}
    };
};

/* specialization */
namespace SIMD
{

#include "avx_fp32.h"
#include "avx_fp64.h"

  template<int ch, class T>
    T broadcast(const T x) { return T::template broadcast<ch>(x); }
  template<int N, class T>
    void transpose(T *x) { T::template transpose<N>(x); }
};

/* vector loads & stores */

template<class T>
inline SIMD::vector<T> VLDA(const T &x) {return SIMD::vector<T>::aload(x);}
template<class T>
inline SIMD::vector<T> VLDU(const T &x) {return SIMD::vector<T>::uload(x);}
template<class T>
inline SIMD::vector<T>& VREF(const T &x) {return (SIMD::vector<T>&)x;}


