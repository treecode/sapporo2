#pragma once

template<>
struct vector<double>
{
  typedef float  _v4sf  __attribute__((vector_size(16)));
  typedef double _v2df  __attribute__((vector_size(16)));
  enum {WIDTH = 2};
  _v2df val;
  vector() {}
  vector(_v2df _val) : val(_val) {};
  vector(const double  f) : val((_v2df){f,f}) {}
  vector(const double  a, const double b) : val((_v2df){a,b}) {}
  vector(const double *p, const bool aligned = true) : 
    val(aligned ? *(_v2df*)p : __builtin_ia32_loadupd(p)) {}

  operator _v2df() const {return val;}

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

  const vector operator!=(const vector a) const 
  {
    return vector(__builtin_ia32_cmpneqpd(val, a.val));
  }
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

  //Added by JB
  const vector operator&(const _v4sf &rhs) const{
    return __builtin_ia32_andpd(val, (_v2df)rhs);
  }
  

  operator bool() const {return true;}

  //JB removed const
  double operator[] (const int i) const {
    switch(i) {
      case 0:	return __builtin_ia32_vec_ext_v2df(val, 0);
      case 1:	return __builtin_ia32_vec_ext_v2df(val, 1);
      default: assert(0);
    }
  }



  inline static void transpose(const vector v0, const vector v1, vector &t0, vector &t1)
  {
    t0 = __builtin_ia32_unpcklpd(v0, v1);
    t1 = __builtin_ia32_unpckhpd(v0, v1);
  }

  template<const int N>
    inline static void transpose(vector *d)
    {
      vector v1, v2;
      switch(N)
      {
        case 1:
          transpose(d[0], d[1], d[0], d[1]);
          break;
        case 2:
          transpose(d[0], d[2], d[0], d[2]);
          transpose(d[5], d[7], d[5], d[7]);
          transpose(d[1], d[3], v1, v2);
          transpose(d[4], d[6], d[1], d[3]);
          d[4] = v1;
          d[6] = v2;
          break;
        default:
          assert(N == 1);
      }
    }


  inline static void sstore(double &ptr, const vector val)
  {
    __builtin_ia32_movntpd(&ptr, val);
  }

  inline static vector astore(double &ptr, const vector val)
  {
    *(_v2df*)&ptr = val;
    return val;
  }
  inline static vector ustore(double &ptr, const vector val)
  {
    __builtin_ia32_storeupd(&ptr, val);
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
        N == 0 ? (0 << 0) + (0 << 0) : (1 << 0) + (1 << 1);
      return __builtin_ia32_shufpd(x, x, mask);
    }

  inline static vector sqrt(const vector x)
  {
    return __builtin_ia32_sqrtpd(x);
  }

  inline static vector rsqrt(const vector x)
  {
#if 1
    const _v2df y = __builtin_ia32_cvtps2pd(__builtin_ia32_rsqrtps(__builtin_ia32_cvtpd2ps(x)));
    return (vector(-0.5) * y) * (x*y*y + vector(-3.0));
#else
    const _v4sf y = __builtin_ia32_cvtpd2ps(x);
    const _v4sf z = __builtin_ia32_rsqrtps(y);
    return __builtin_ia32_cvtps2pd(z);
#endif
  }



};
