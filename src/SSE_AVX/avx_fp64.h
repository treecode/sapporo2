#pragma once

template<>
struct vector<double>
{
  typedef float  _v8sf  __attribute__((vector_size(32)));
  typedef double _v4df  __attribute__((vector_size(32)));
  typedef float  _v4sf  __attribute__((vector_size(16)));
  typedef double _v2df  __attribute__((vector_size(16)));

  enum {WIDTH = 4};
  _v4df val;
  vector() {}
  vector(_v4df _val) : val(_val) {};
  vector(const double  f) : val((_v4df){f,f,f,f}) {}
  vector(const double  a, const double b, const double c, const double d) : 
    val((_v4df){a,b,c,d}) {}
  vector(const double *p, const bool aligned = true) : 
    val(aligned ? *(_v4df*)p : __builtin_ia32_loadupd256(p)) {}

  operator _v4df() const {return val;}

  const vector& operator =(const vector a) {val  = a.val; return *this;}
  const vector& operator+=(const vector a) {val += a.val; return *this;}
  const vector& operator-=(const vector a) {val -= a.val; return *this;}
  const vector& operator*=(const vector a) {val *= a.val; return *this;}
  const vector& operator/=(const vector a) {val /= a.val; return *this;}
  const vector operator -()  const {return vector(-val);}
  const vector operator +(const vector a) const {return vector(val + a.val);}
  const vector operator -(const vector a) const {return vector(val - a.val);}
  const vector operator *(const vector a) const {return vector(val * a.val);}
  const vector operator /(const vector a) const {return vector(val / a.val);}

  operator bool() const {return true;}
  const vector operator!=(const vector a) const 
  {
    return vector(__builtin_ia32_cmppd256(val, a.val, 28));
  }
#if 0
  const vector operator==(const vector a) const {
    return vector(__builtin_ia32_cmpeqpd(val, a.val));
  }

  const vector operator<(const vector &rhs) const{
    return __builtin_ia32_cmpltpd(val, rhs.val);
  }
  const vector operator<=(const vector &rhs) const{
    return __builtin_ia32_cmplepd(val, rhs.val);
  }
  const vector operator>(const vector &rhs) const{
    return __builtin_ia32_cmpgtpd(val, rhs.val);
  }
  const vector operator>=(const vector &rhs) const{
    return __builtin_ia32_cmpgepd(val, rhs.val);
  }
  const vector operator|(const vector &rhs) const{
    return __builtin_ia32_orpd(val, rhs.val);
  }
  const vector operator&(const vector &rhs) const{
    return __builtin_ia32_andpd(val, rhs.val);
  }
#endif


  const double operator[] (const int i) const {
    union {_v4df v; double s[4];} test;
    test.v = val;
    return test.s[i];
#if 0
    switch(i) {
      case 0:	return __builtin_ia32_vec_ext_v4df(val, 0);
      case 1:	return __builtin_ia32_vec_ext_v4df(val, 1);
      case 2:	return __builtin_ia32_vec_ext_v4df(val, 2);
      case 3:	return __builtin_ia32_vec_ext_v4df(val, 3);
      default: assert(0);
    }
#endif
  }



  inline static void transpose(
      const vector v0, const vector v1, const vector v2, const vector v3,
      vector &t0, vector &t1, vector &t2, vector &t3)
  {
#if 0
    const _v4df a0 = __builtin_ia32_unpcklpd256(v0, v2);
    const _v4df a1 = __builtin_ia32_unpckhpd256(v0, v2);
    const _v4df a2 = __builtin_ia32_unpcklpd256(v1, v3);
    const _v4df a3 = __builtin_ia32_unpckhpd256(v1, v3);

    t0 = __builtin_ia32_unpcklpd256(a0, a2);
    t1 = __builtin_ia32_unpckhpd256(a0, a2);
    t2 = __builtin_ia32_unpcklpd256(a1, a3);
    t3 = __builtin_ia32_unpckhpd256(a1, a3);
#else
    const _v4df a0 = __builtin_ia32_unpcklpd256(v0, v1);
    const _v4df a1 = __builtin_ia32_unpckhpd256(v0, v1);
    const _v4df a2 = __builtin_ia32_unpcklpd256(v2, v3);
    const _v4df a3 = __builtin_ia32_unpckhpd256(v2, v3);

    t0 = __builtin_ia32_vperm2f128_pd256(a0, a2, 0x20);
    t1 = __builtin_ia32_vperm2f128_pd256(a1, a3, 0x20);
    t2 = __builtin_ia32_vperm2f128_pd256(a0, a2, 0x31);
    t3 = __builtin_ia32_vperm2f128_pd256(a1, a3, 0x31);
#endif
  }

  template<const int N>
    inline static void transpose(vector *d)
    {
      if (N == 1)
      {
        transpose(d[0], d[1], d[2], d[3], d[0], d[1], d[2], d[3]);
      }
      else if (N == 2)
      {
        const vector m11[4] = {d[0], d[ 2], d[ 4], d[ 6]};
        const vector m12[4] = {d[1], d[ 3], d[ 5], d[ 7]};
        const vector m21[4] = {d[8], d[10], d[12], d[14]};
        const vector m22[4] = {d[9], d[11], d[13], d[15]};
        transpose(m11[0], m11[1], m11[2], m11[3], d[0], d[ 2], d[ 4], d[ 6]);
        transpose(m21[0], m21[1], m21[2], m21[3], d[1], d[ 3], d[ 5], d[ 7]);
        transpose(m12[0], m12[1], m12[2], m12[3], d[8], d[10], d[12], d[14]);
        transpose(m22[0], m22[1], m22[2], m22[3], d[9], d[11], d[13], d[15]);
      }
      else
        assert(N == 1);
    }

  inline static void sstore(double &ptr, const vector val)
  {
    __builtin_ia32_movntpd256(&ptr, val);
  }

  inline static vector astore(double &ptr, const vector val)
  {
    *(_v4df*)&ptr = val;
    return val;
  }
  inline static vector ustore(double &ptr, const vector val)
  {
    __builtin_ia32_storeupd256((double*)&ptr, val);
    return val;
  }

  inline static vector uload(const double &ptr)
  {
    return vector(&ptr, false);
  }
  inline static vector aload(const double &ptr)
  {
    return vector(&ptr, true);
  }

  inline static vector& aref(double &ref) 
  {
    return (vector&)ref;
  }

  template<const int N>
    inline static vector broadcast(const vector x)
    {
      const int mask = 
        N == 0 ? (0 << 0) + (0 << 1) + (0 << 2) + (0 << 3) :
        N == 1 ? (1 << 0) + (1 << 1) + (0 << 2) + (0 << 3) :
        N == 2 ? (0 << 0) + (0 << 1) + (0 << 2) + (0 << 3) :
        (0 << 0) + (0 << 1) + (1 << 2) + (1 << 3);

      const vector y = __builtin_ia32_shufpd256(x, x, mask);
      if (N < 2) return __builtin_ia32_vperm2f128_pd256(y, y, (0 << 0) + (0 << 4));
      else       return __builtin_ia32_vperm2f128_pd256(y, y, (1 << 0) + (1 << 4));
    }

  inline static vector sqrt(const vector x)
  {
    return __builtin_ia32_sqrtpd256(x);
  }

  inline static vector rsqrt(const vector x)
  {
#if 1
    const _v4df y = __builtin_ia32_cvtps2pd256(__builtin_ia32_rsqrtps(__builtin_ia32_cvtpd2ps256(x)));
    return (vector(-0.5) * y) * (x*y*y + vector(-3.0));
#else
    const _v4sf y = __builtin_ia32_cvtpd2ps256(x);
    const _v4sf z = __builtin_ia32_rsqrtps(y);
    return __builtin_ia32_cvtps2pd256(z);
#endif
  }

};
