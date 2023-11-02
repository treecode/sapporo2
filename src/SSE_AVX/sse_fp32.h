#pragma once

template<>
struct vector<float>
{
  typedef float  _v4sf  __attribute__((vector_size(16)));

  enum {WIDTH = 4};
  _v4sf val;
  vector() {}
  vector(_v4sf _val) : val(_val) {};
  vector(const float  f) : val((_v4sf){f,f,f,f}) {}
  vector(const float  a, const float b, const float c, const float d) : val((_v4sf){a,b,c,d}) {}
  vector(const float *p, const bool aligned = true) : 
    val(aligned ? *(_v4sf*)p : __builtin_ia32_loadups(p)) {}

  operator _v4sf() const {return val;}

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
    return vector(__builtin_ia32_cmpneqps(val, a.val));
  }
  const vector operator==(const vector a) const {
    return vector(__builtin_ia32_cmpeqps(val, a.val));
  }

  const vector operator<(const vector &rhs) const{
    return __builtin_ia32_cmpltps(val, rhs.val);
  }
  const vector operator<=(const vector &rhs) const{
    return __builtin_ia32_cmpleps(val, rhs.val);
  }
  const vector operator>(const vector &rhs) const{
    return __builtin_ia32_cmpgtps(val, rhs.val);
  }
  const vector operator>=(const vector &rhs) const{
    return __builtin_ia32_cmpgeps(val, rhs.val);
  }
  const vector operator|(const vector &rhs) const{
    return __builtin_ia32_orps(val, rhs.val);
  }
  const vector operator&(const vector &rhs) const{
    return __builtin_ia32_andps(val, rhs.val);
  }

  operator bool() const {return true;}

  //edit JB
  float operator[] (const int i) const {
    switch(i) {
      case 0:	return __builtin_ia32_vec_ext_v4sf(val, 0);
      case 1:	return __builtin_ia32_vec_ext_v4sf(val, 1);
      case 2:   return __builtin_ia32_vec_ext_v4sf(val, 2);
      case 3:   return __builtin_ia32_vec_ext_v4sf(val, 3);
      default: assert(0);
    }
  }



  inline static void transpose(
      const vector v0, const vector v1, const vector v2, const vector v3,
      vector &t0, vector &t1, vector &t2, vector &t3)
  {
    t0 = __builtin_ia32_unpcklps(v0, v1);
    t1 = __builtin_ia32_unpckhps(v0, v1);

    const _v4sf a0 = __builtin_ia32_unpcklps(v0, v2);
    const _v4sf a1 = __builtin_ia32_unpckhps(v0, v2);
    const _v4sf a2 = __builtin_ia32_unpcklps(v1, v3);
    const _v4sf a3 = __builtin_ia32_unpckhps(v1, v3);

    t0 = __builtin_ia32_unpcklps(a0, a2);
    t1 = __builtin_ia32_unpckhps(a0, a2);
    t2 = __builtin_ia32_unpcklps(a1, a3);
    t3 = __builtin_ia32_unpckhps(a1, a3);
  }

  template<const int N>
    inline static void transpose(vector *d)
    {
      if (N == 1)
        transpose(d[0], d[1], d[2], d[3], d[0], d[1], d[2], d[3]);
      else if (N==2)
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
        assert(N==1);
    }

  inline static void sstore(float &ptr, const vector val)
  {
    __builtin_ia32_movntps(&ptr, val);
  }

  inline static vector astore(float &ptr, const vector val)
  {
    *(_v4sf*)&ptr = val;
    return val;
  }
  inline static vector ustore(float &ptr, const vector val)
  {
    __builtin_ia32_storeups((float*)&ptr, val);
    return val;
  }

  inline static vector aload(const float &ptr)
  {
    return vector(&ptr, true);
  }
  inline static vector uload(const float &ptr)
  {
    return vector(&ptr, false);
  }
  inline static vector& aref(float &ptr)
  {
    return (vector&)ptr;
  }

  template<const int N>
    inline static vector broadcast(const vector x)
    {
      return
        N == 0 ? __builtin_ia32_shufps(x,x, 0x00) : 
        N == 1 ? __builtin_ia32_shufps(x,x, 0x55) :
        N == 2 ? __builtin_ia32_shufps(x,x, 0xAA) :
        __builtin_ia32_shufps(x,x, 0xFF);
    }


  inline static vector sqrt(const vector x)
  {
    return __builtin_ia32_sqrtps(x);
  }

  inline static vector rsqrt(const vector x)
  {
#if 1
    const vector y = __builtin_ia32_rsqrtps(x);
    return (vector(-0.5f) * y) * (x*y*y + vector(-3.0f));
#else
    return __builtin_ia32_rsqrtps(x);
#endif
  }
};


