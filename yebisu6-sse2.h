#ifndef YEBISU6_SSE2_H
#define YEBISU6_SSE2_H
//// h6-grape_sse2.h         ////
//// based on yebisu6-sse2.h ////
//// modeified for BRIDGE    ////
#include <assert.h>
// #include <omp.h>
// #include "coord.h"
#include "accum.h"
//#include "v4df.h"
//#include "v4sf.h"

//#include "nbody_particle_MPI.h"

struct Force{
    //static const int nword = 9;
    vec acc;
    vec jrk;
    vec snp;
    double phi;
    int nnb_id;
    float  nnb_r2;
    Force() : acc(0.0), jrk(0.0), snp(0.0), phi(0.0), nnb_id(-1), nnb_r2(HUGE){}    void operator += (Force &rhs){
        acc += rhs.acc;
        jrk += rhs.jrk;
        snp += rhs.snp;
        phi += rhs.phi;
    }
    void clear(){
        acc[0] = acc[1] = acc[2] = 0.0;
        jrk[0] = jrk[1] = jrk[2] = 0.0;
        snp[0] = snp[1] = snp[2] = 0.0;
        phi = 0.0;
        nnb_id = -1;
        nnb_r2 = HUGE;

    }
  //// for BH particles ////
    void add_BH_force(double eps2, double mBH, vec pos_BH, vec vel_BH, vec acc_BH,
		      vec pos_i, vec vel_i, vec acc_i, int iBH){

            vec pos0 = pos_BH - pos_i;
            vec vel0 = vel_BH - vel_i;
            vec acc0 = acc_BH - acc_i;
            double dist2 = pos0*pos0;
	    double r2 = dist2 + eps2;
            double rinv2 = 1.0 / r2;
            double rinv = sqrt(rinv2);
            double rinv3 = rinv*rinv2;

            double alpha = rinv2*(vel0*pos0);
            double beta = rinv2*(vel0*vel0 + pos0*acc0) + alpha*alpha;

            vec acc1 = mBH * rinv3 * pos0;
            vec jerk1 = mBH*rinv3*vel0 - 3.0*alpha*acc1;
            double phi1 = -mBH * rinv;

            snp += mBH*rinv3*acc0 - 6.0*alpha*jerk1 - 3.0*beta*acc1;
            acc += acc1;
            jrk += jerk1;
            phi += phi1;

            if(dist2<nnb_r2){
                nnb_id = iBH;
		nnb_r2 = dist2;
                //cout << index << endl;
            }
    }
    
};

namespace yebisu{
  //namespace grape{
	typedef double v2df __attribute__ ((vector_size(16)));
	struct predictor{
		double pos [3]; //  6
		double vel [3]; // 12
		double acc [3]; // 18
		double mass;    // 20
		double id  ;    // 22
	    //double pad [2];
	    double eps2;
		predictor(){
			assert(sizeof(*this) == 12*8);
		}
	};
	struct predictor2{
		double pos [3][2]; //  6
		double vel [3][2]; // 12
		double acc [3][2]; // 18
		double mass[2];    // 20
		double id  [2];    // 22
	    //double pad [2];
	    double eps2[2];
		predictor2(){
			assert(sizeof(*this) == 24*8);
		}
	};
	struct particle{
		double pos [3][2]; //  6
		double vel [3][2]; // 12
		double acc [3][2]; // 18
		double jrk [3][2]; // 24
		double snp [3][2]; // 30
		double crk [3][2]; // 36
		double mass[2];    // 38
		double time[2];    // 40
		double id  [2];    // 42
	    //double pad [6];    // 48
	    double eps2[2];
		particle(){
			assert(sizeof(*this) ==  44*8);
		}
	};
/*
	struct force{
		double acc[3];
		double jrk[3];
		double snp[3];
		double pot;
		int    nnb_id;
		float  nnb_r2;

		void clear(){
			acc[0] = acc[1] = acc[2] = 0.0;
			jrk[0] = jrk[1] = jrk[2] = 0.0;
			snp[0] = snp[1] = snp[2] = 0.0;
			pot = 0.0;
			nnb_id = -1;
			nnb_r2 = HUGE;
		}
	};
*/
	struct vpredictor{
		v2df pos [3]; //  6
		v2df vel [3]; // 12
		v2df acc [3]; // 18
		v2df mass;    // 20
		v2df id;      // 22
	    //v2df pad;     // 24
	    v2df eps2;
		vpredictor(){
			assert(sizeof(*this) == 24*8);
		}
	};
	struct vparticle{
		v2df pos [3]; //  6
		v2df vel [3]; // 12
		v2df acc [3]; // 18
		v2df jrk [3]; // 24
		v2df snp [3]; // 30
		v2df crk [3]; // 36
		v2df mass;    // 38
		v2df time;    // 40
		v2df id;      // 42
	    //v2df pad[3];  // 48
	    v2df eps2;
		vparticle(){
			assert(sizeof(*this) ==  44*8);
		}
	};

	class Yebisu{
	  //class Grape{
		int nbody, n;
		int n_array;
		int njblocks;
		static const int JBSIZE = 16384;
		double dtmax, invdtmax;
		particle  *ptcl;
		predictor2 *pred2;
		vparticle  *vptcl;
		vpredictor *vpred;
		//double *dtime;
		double frac_time(double t){
			t -= dtmax * int(t*invdtmax);
			return t;
		}
	public:
		//force *fobuf;
		//Yebisu(unsigned _nbody, double _dtmax) :
		//Grape(int _nbody, double _dtmax) :
		
		Yebisu(int _nbody, double _dtmax) :
			nbody(_nbody), dtmax(_dtmax)
		{
		    //assert(nbody%2 == 0);
		    nbody = _nbody;
		    n = nbody;
                    if(nbody%2 != 0){
			n++;
		    }
		    assert(n%2 == 0);
		    njblocks = n / JBSIZE + (nbody % JBSIZE ? 1 : 0);
		    invdtmax = 1./dtmax;
		    ptcl = new particle[n/2];
		    pred2 = new predictor2[n/2];
		    //fobuf = new force[nbody];			
		    assert((unsigned long)(ptcl) % 16 == 0);
		    assert((unsigned long)(pred2) % 16 == 0);
		    vptcl = (vparticle  *)ptcl;
		    vpred = (vpredictor *)pred2;
		}
		
		Yebisu()
		//Grape()
		{		  
		  //n_array = 131072;
		  n_array = 65536;		  
		    dtmax = 1.0;
		    invdtmax = 1./dtmax;
		    ptcl = new particle[n_array/2];
		    pred2 = new predictor2[n_array/2];
		    assert((unsigned long)(ptcl) % 16 == 0);
		    assert((unsigned long)(pred2) % 16 == 0);
		    vptcl = (vparticle  *)ptcl;
		    vpred = (vpredictor *)pred2;
		    for(int j=0;j<(int)(n_array/2);j++){
                        for(int k=0;k<2;k++){
			    ptcl[j].id[k] = -1;
			    ptcl[j].mass[k] = 0.0;
                        }			
                    }
		    cerr << "construct:yebisu-h6-sse2" << endl;
		}
		~Yebisu(){
		//~Grape(){
		  //if(ptcl!=NULL) delete [] ptcl;
		  //if(pred2!=NULL) delete [] pred2;
			//delete [] fobuf;		
		}
		
		 //// Call for initialization at the beginning of the step ////
                void initialize(int _n)
                {
                    nbody = _n;
		    n = nbody;
                    if(nbody%2 != 0){
			n++;
		    }
		    //n_array = 131072;
		    //if(ptcl==NULL){
		    //ptcl = new particle[n_array/2];		      
		    //}
		    //if(pred2==NULL){
		    //pred2 = new predictor2[n_array/2];		      
		    //} 
		    //vptcl = (vparticle *)ptcl;
		    //vpred = (vpredictor *)pred2;
		    assert(n%2 == 0);
		    njblocks = n / JBSIZE + (n % JBSIZE ? 1 : 0);
                    if(n>n_array){
                        delete [] ptcl;
                        delete [] pred2;
                        //delete [] dtime;
                        ptcl = new particle[n/2];
			pred2 = new predictor2[n/2];
                        assert((unsigned long)(ptcl) % 16 == 0);
			assert((unsigned long)(pred2) % 16 == 0);
                        vptcl = (vparticle  *)ptcl;
			vpred = (vpredictor *)pred2;
                        n_array = n;
			for(int j=0;j<(int)(n_array/2);j++){
			  for(int k=0;k<2;k++){
			    ptcl[j].id[k] = -1;
			    ptcl[j].mass[k] = 0.0;
			  }			
			}
                    } 
		    //cerr << ptcl[0].id[0] << endl;	    
		    //cerr << "initialize yebisu-h6; n = " << n << endl;
                }

		void set_jp(
				int addr,				
				double pos[3], 
				double vel[3], 
				double acc[3], 
				double jrk[3], 
				double snp[3], 
				double crk[3], 
				double mass,
				double time,
				int    id,
				double eps2
				/*nbody_particle_cluster &p*/)
		{
			assert(addr < nbody);
			unsigned ah = addr/2;
			unsigned al = addr%2;
			//vec pos = p.get_pos();
                        //vec vel = p.get_vel();
                        //vec acc = p.get_acc();
                        //vec jrk = p.get_jerk();
                        //vec snp = p.get_snap();
                        //vec crk = p.get_crac();
			//cout << addr << "  " << crk << endl;
			for(int k=0; k<3; k++){
			    ptcl[ah].pos[k][al] = pos[k];
			    ptcl[ah].vel[k][al] = vel[k];
			    ptcl[ah].acc[k][al] = acc[k];
			    ptcl[ah].jrk[k][al] = jrk[k];
			    ptcl[ah].snp[k][al] = snp[k];
			    ptcl[ah].crk[k][al] = crk[k];
			}

			//ptcl[ah].time[al] = p.get_t();//frac_time(time);
			ptcl[ah].time[al] = time;
			/*
			if(p.get_real_index()<0){
			    ptcl[ah].mass[al] = 0.0;			    
			}else{
			  ptcl[ah].mass[al] = p.get_mass();
			}
			ptcl[ah].id[al] = p.get_index();
			ptcl[ah].eps2[al] = p.get_eps2();
			*/
			ptcl[ah].mass[al] = mass;
			ptcl[ah].eps2[al] = eps2;
			ptcl[ah].id[al] = id;
		}

	private:
		inline void predict_one(
				const vparticle &p,
				vpredictor &pr,
				v2df dt)
		{
		    //v2df dt = (ti - p.time);
			v2df dt2 = dt * (v2df){0.5, 0.5};
			v2df dt3 = dt * (v2df){1./3., 1./3.};
			v2df dt4 = dt * (v2df){1./4., 1./4.};
			v2df dt5 = dt * (v2df){1./5., 1./5.};
			for(int k=0; k<3; k++){
				pr.pos[k] = p.pos[k] + dt  * (
				             p.vel[k] + dt2 * (
				              p.acc[k] + dt3 * (
				               p.jrk[k] + dt4 * (
				                p.snp[k] + dt5 * (
				                 p.crk[k]    )))));
				pr.vel[k] = p.vel[k] + dt  * (
							 p.acc[k] + dt2 * (
							  p.jrk[k] + dt3 * (
							   p.snp[k] + dt4 * (
								p.crk[k]     ))));
				pr.acc[k] = p.acc[k] + dt  * (
							 p.jrk[k] + dt2 * (
							  p.snp[k] + dt3 * (
							   p.crk[k]      )));
			}
			pr.id   = p.id;
			pr.mass = p.mass;
			pr.eps2 = p.eps2;
		}
	public:
		void predict_all(double tsys, int nj){
		    //double  tid = frac_time(tsys);
		    //if((tid==0) && (tsys!=0)) tid = dtmax;
		    //v2df ti = {tid, tid};
		  if(nj>n){ 
		    cerr << "Error nj>n" << endl; 
		    exit(1);
		  }
		  if(nj%2 != 0) nj++;
#pragma omp parallel for
		    for(int i=0; i<int(nj/2); i++){
			double dt0 = tsys - ptcl[i].time[0];
			double dt1 = tsys - ptcl[i].time[1];
			v2df dt = {dt0, dt1};
			predict_one(vptcl[i], vpred[i], dt);
		    }
		}

		void no_predict_all(int nj){
		  if(nj>n){ 
		    cerr << "Error nj>n" << endl; 
		    exit(1);
		  }
		   if(nj%2 != 0) nj++;
#pragma omp parallel for
                        for(int i=0; i<int(nj/2); i++){
                            for(int k=0; k<3; k++){
                                vpred[i].pos[k] = vptcl[i].pos[k];
				vpred[i].vel[k] = vptcl[i].vel[k];
                                vpred[i].acc[k] = vptcl[i].acc[k];
                            }
                            vpred[i].mass = vptcl[i].mass;
                            vpred[i].id = vptcl[i].id;
			    vpred[i].eps2 = vptcl[i].eps2;
                        }
			
                }
/*
		void pick_up_predictors(
                                int ni,
                                const int index[],
                                predictor pred[])
                {
                        for(int i=0; i<ni; i++){
                            int addr = index[i];
                            unsigned ah = addr/2;
                            unsigned al = addr%2;
                            for(int k=0; k<3; k++){
                                pred[i].pos[k] = pred2[ah].pos[k][al];
				pred[i].vel[k] = pred2[ah].vel[k][al];
                                pred[i].acc[k] = pred2[ah].acc[k][al];

                            }
                            pred[i].mass = pred2[ah].mass[al];
                            pred[i].id = pred2[ah].id[al];
                        }
                }
*/
		void pick_up_predictor(int addr,
                                       predictor &pred)
                {
                    unsigned ah = addr/2;
                    unsigned al = addr%2;
                    for(int k=0; k<3; k++){
                        pred.pos[k] = pred2[ah].pos[k][al];
                        pred.vel[k] = pred2[ah].vel[k][al];
                        pred.acc[k] = pred2[ah].acc[k][al];
                    }
                    pred.mass = pred2[ah].mass[al];
                    pred.id = pred2[ah].id[al];
		    pred.eps2 = pred2[ah].eps2[al];
		}		
		/*
		void get_predictor(int addr,
				   vec &pos,
				   vec &vel,
				   vec &acc)
                {
                    //int addr = index[i];
                    unsigned ah = addr/2;
                    unsigned al = addr%2;
                    for(int k=0; k<3; k++){
                        pos[k] = pred2[ah].pos[k][al];
                        vel[k] = pred2[ah].vel[k][al];
                        acc[k] = pred2[ah].acc[k][al];
                    }
                }
		
		vec get_predicted_pos(int addr)
                {
                    vec pos;
                    unsigned ah = addr/2;
                    unsigned al = addr%2;
                    for(int k=0;k<3;k++){
                        pos[k] = pred2[ah].pos[k][al];
                    }
                    return pos;
                }
		*/
		void transfer_j_particles(int nj){
		}
	private:
		inline double v2df_sum(v2df v){
			return __builtin_ia32_vec_ext_v2df(v, 0)
			     + __builtin_ia32_vec_ext_v2df(v, 1);
		}

		inline void calc_force_on_i(
		    //int i,
		    Force &fobuf,
				const v2df posi[3],
				const v2df veli[3],
				const v2df acci[3],
				const v2df eps2,
				const v2df idi,
				unsigned js,
				unsigned je)
		{
		    		    
#if 0
			v2df zero = {0.0, 0.0};
			DPaccum <v2df, v2df> ax(zero, zero), ay(zero, zero), 
			    az(zero, zero), mpot(zero, zero);
#else
			v2df Ax={0.,0.}, Ay={0.,0.}, Az={0.,0.}, mPot={0.,0.};
#endif
			v2df Jx={0.,0.}, Jy={0.,0.}, Jz={0.,0.};
			v2df Sx={0.,0.}, Sy={0.,0.}, Sz={0.,0.};

			v2df nnb = {HUGE, HUGE};   			

			// for(unsigned j=0; j<nbody/4; j++){
			for(unsigned j=js; j<je; j++){
				// __builtin_prefetch(&vpred[j+1], 0, 1);
				const vpredictor &pj = vpred[j];
				//v2df mask = (v2df)__builtin_ia32_cmpneqpd(idi, pj.id);
				v2df mask = __builtin_ia32_andpd(
				    (v2df)__builtin_ia32_cmpneqpd(idi, pj.id), 
				    (v2df)__builtin_ia32_cmpgepd(pj.id, (v2df){0x00000000U,0x00000000U}));
				v2df bias = __builtin_ia32_andnpd(mask, (v2df){HUGE, HUGE});
				//(~mask)&HUGE
				v2df dx  = pj.pos[0] - posi[0];
				v2df dy  = pj.pos[1] - posi[1];
				v2df dz  = pj.pos[2] - posi[2];
				v2df dvx = pj.vel[0] - veli[0];
				v2df dvy = pj.vel[1] - veli[1];
				v2df dvz = pj.vel[2] - veli[2];
				v2df dax = pj.acc[0] - acci[0];
				v2df day = pj.acc[1] - acci[1];
				v2df daz = pj.acc[2] - acci[2];

				//v2df eps2 = eps2i;
				//v2df eps2 = __builtin_ia32_maxpd(eps2i, pj.eps2);
				
				v2df r2 = dx*dx + dy*dy + dz*dz + eps2 + bias;
/*
				if((int)__builtin_ia32_vec_ext_v2df(idi, 0)==188){
				    v2df tmp = eps2i;
				    cout << (int)__builtin_ia32_vec_ext_v2df(idi, 0) << "  "<< j*2 << "  " << __builtin_ia32_vec_ext_v2df(tmp, 0) << " " << j*2+1 << "  " << __builtin_ia32_vec_ext_v2df(tmp, 1) << endl;}
*/
				for(unsigned i=0;i<2;i++){
				    double r2_tmp = __builtin_ia32_vec_ext_v2df(r2, i);
				    if(r2_tmp < __builtin_ia32_vec_ext_v2df(nnb, 0)){
					v2df tmp = {r2_tmp, __builtin_ia32_vec_ext_v2df(pj.id, i)};
					nnb = tmp;
				    }
				}
	
				v2df rinv2 = (v2df){1.0, 1.0} / r2;
				v2df rinv  = __builtin_ia32_sqrtpd(rinv2);
				
				rinv  = __builtin_ia32_andpd(rinv, mask);
				
				v2df rv = dx *dvx + dy *dvy + dz *dvz;
				v2df v2 = dvx*dvx + dvy*dvy + dvz*dvz;
				v2df ra = dx *dax + dy *day + dz *daz;
				// v4sf rinv  = r2.rsqrt();
				v2df alpha = rv * rinv2;
				v2df beta  = (v2 + ra) * rinv2 + alpha * alpha;
				rinv *= pj.mass;
				mPot += rinv;
				v2df rinv3 = rinv * rinv2;

				v2df ax = rinv3 * dx;
				v2df ay = rinv3 * dy;
				v2df az = rinv3 * dz;
				alpha *= (v2df){-3.0, -3.0};
				v2df jx = rinv3 * dvx + alpha * ax;
				v2df jy = rinv3 * dvy + alpha * ay;
				v2df jz = rinv3 * dvz + alpha * az;
				alpha *= (v2df){2.0, 2.0};
				beta  *= (v2df){-3.0, -3.0};
				v2df sx = rinv3 * dax + alpha * jx + beta * ax;
				v2df sy = rinv3 * day + alpha * jy + beta * ay;
				v2df sz = rinv3 * daz + alpha * jz + beta * az;

				Ax += ax;
				Ay += ay;
				Az += az;
				Jx += jx;
				Jy += jy;
				Jz += jz;
				Sx += sx;
				Sy += sy;
				Sz += sz;
				//cout << (int)__builtin_ia32_vec_ext_v2df(idi, 0) << "  "<< j << "  " << __builtin_ia32_vec_ext_v2df(Ax, 0) << endl;
			}
			fobuf.acc[0] += v2df_sum(v2df(Ax));
			fobuf.acc[1] += v2df_sum(v2df(Ay));
			fobuf.acc[2] += v2df_sum(v2df(Az));
			fobuf.jrk[0] += v2df_sum(Jx);
			fobuf.jrk[1] += v2df_sum(Jy);
			fobuf.jrk[2] += v2df_sum(Jz);
			fobuf.snp[0] += v2df_sum(Sx);
			fobuf.snp[1] += v2df_sum(Sy);
			fobuf.snp[2] += v2df_sum(Sz);
			fobuf.phi    += -v2df_sum(mPot);
			
			//cout << (int)__builtin_ia32_vec_ext_v2df(idi, 0)  << "  "<< fobuf.acc[0] << endl;

			fobuf.nnb_r2 = __builtin_ia32_vec_ext_v2df(nnb, 0);
			fobuf.nnb_id = (int)__builtin_ia32_vec_ext_v2df(nnb, 1);
			
		}
                inline void calc_force_on_i_epsj(
                    //int i,
                    Force &fobuf,
		    const v2df posi[3],
		    const v2df veli[3],
		    const v2df acci[3],
		    const v2df idi,
		    unsigned js,
		    unsigned je)
		    {

#if 0
                        v2df zero = {0.0, 0.0};
                        DPaccum <v2df, v2df> ax(zero, zero), ay(zero, zero),
                            az(zero, zero), mpot(zero, zero);
#else
                        v2df Ax={0.,0.}, Ay={0.,0.}, Az={0.,0.}, mPot={0.,0.};
#endif
                        v2df Jx={0.,0.}, Jy={0.,0.}, Jz={0.,0.};
                        v2df Sx={0.,0.}, Sy={0.,0.}, Sz={0.,0.};

                        v2df nnb = {HUGE, HUGE};

                        // for(unsigned j=0; j<nbody/4; j++){
                        for(unsigned j=js; j<je; j++){
			    // __builtin_prefetch(&vpred[j+1], 0, 1);
			    const vpredictor &pj = vpred[j];
			    v2df mask = __builtin_ia32_andpd(
				(v2df)__builtin_ia32_cmpneqpd(idi, pj.id),
				(v2df)__builtin_ia32_cmpgepd(pj.id, (v2df){0x00000000U,0x00000000U}));
			    v2df bias = __builtin_ia32_andnpd(mask, (v2df){HUGE, HUGE});
			    //(~mask)&HUGE
			    v2df dx  = pj.pos[0] - posi[0];
			    v2df dy  = pj.pos[1] - posi[1];
			    v2df dz  = pj.pos[2] - posi[2];
			    v2df dvx = pj.vel[0] - veli[0];
			    v2df dvy = pj.vel[1] - veli[1];
			    v2df dvz = pj.vel[2] - veli[2];
			    v2df dax = pj.acc[0] - acci[0];
			    v2df day = pj.acc[1] - acci[1];
			    v2df daz = pj.acc[2] - acci[2];
			    v2df eps2 = pj.eps2;

			    v2df r2 = dx*dx + dy*dy + dz*dz + eps2 + bias;
			    for(unsigned i=0;i<2;i++){
				double r2_tmp = __builtin_ia32_vec_ext_v2df(r2, i);
				if(r2_tmp < __builtin_ia32_vec_ext_v2df(nnb, 0)){
				    v2df tmp = {r2_tmp, __builtin_ia32_vec_ext_v2df(pj.id, i)};
				    nnb = tmp;
				}
			    }			    
			    
			    v2df rinv2 = (v2df){1.0, 1.0} / r2;
			    v2df rinv  = __builtin_ia32_sqrtpd(rinv2);
			    
			    rinv  = __builtin_ia32_andpd(rinv, mask);
			    
			    v2df rv = dx *dvx + dy *dvy + dz *dvz;
			    v2df v2 = dvx*dvx + dvy*dvy + dvz*dvz;
			    v2df ra = dx *dax + dy *day + dz *daz;
			    // v4sf rinv  = r2.rsqrt();
			    v2df alpha = rv * rinv2;
			    v2df beta  = (v2 + ra) * rinv2 + alpha * alpha;
			    rinv *= pj.mass;
			    mPot += rinv;
			    v2df rinv3 = rinv * rinv2;
			    
			    v2df ax = rinv3 * dx;
			    v2df ay = rinv3 * dy;
			    v2df az = rinv3 * dz;
			    alpha *= (v2df){-3.0, -3.0};
			    v2df jx = rinv3 * dvx + alpha * ax;
			    v2df jy = rinv3 * dvy + alpha * ay;
			    v2df jz = rinv3 * dvz + alpha * az;
			    alpha *= (v2df){2.0, 2.0};
			    beta  *= (v2df){-3.0, -3.0};
			    v2df sx = rinv3 * dax + alpha * jx + beta * ax;
			    v2df sy = rinv3 * day + alpha * jy + beta * ay;
			    v2df sz = rinv3 * daz + alpha * jz + beta * az;
			    
			    Ax += ax;
			    Ay += ay;
			    Az += az;
			    Jx += jx;
			    Jy += jy;
			    Jz += jz;
			    Sx += sx;
			    Sy += sy;
			    Sz += sz;
			    
			}
			fobuf.acc[0] += v2df_sum(v2df(Ax));
			fobuf.acc[1] += v2df_sum(v2df(Ay));
			fobuf.acc[2] += v2df_sum(v2df(Az));
			fobuf.jrk[0] += v2df_sum(Jx);
			fobuf.jrk[1] += v2df_sum(Jy);
			fobuf.jrk[2] += v2df_sum(Jz);
			fobuf.snp[0] += v2df_sum(Sx);
			fobuf.snp[1] += v2df_sum(Sy);
			fobuf.snp[2] += v2df_sum(Sz);
			fobuf.phi    += -v2df_sum(mPot);
			
			fobuf.nnb_r2 = __builtin_ia32_vec_ext_v2df(nnb, 0);
			fobuf.nnb_id = (int)__builtin_ia32_vec_ext_v2df(nnb, 1);
			
		    }
	public:
		void calc_force_on_predictors(
                                int ni,
                                //predictor pred[],
                                std::vector<predictor> &pred,
                                //Force force[],
                                std::vector<Force> &force,
                                //float eps2[])
                                std::vector<float> &eps2)
                {
			for(unsigned jb=0; jb<(unsigned)njblocks; jb++){
			    unsigned nbody2 = n / 2;
			    unsigned js = ((jb  )*nbody2) / njblocks;
			    unsigned je = ((jb+1)*nbody2) / njblocks;
				//for(int js=js0; js<nj2; js+= JBSIZE/2){
				//int je = std::min(nj2-js, JBSIZE/2);
			    //if(je>n) je = n;
			    //cerr << "js=" << js << " je=" << je << endl;
#pragma omp parallel for
                            for(int i=0; i<ni; i++){
                                /*
                                unsigned ih = i/2;
				unsigned il = i%2;
				v2df posi[3] = {
				    {pred[ih].pos[0][il], pred[ih].pos[0][il]},
				    {pred[ih].pos[1][il], pred[ih].pos[1][il]},
				    {pred[ih].pos[2][il], pred[ih].pos[2][il]},
				};
				v2df veli[3] = {
				    {pred[ih].vel[0][il], pred[ih].vel[0][il]},
				    {pred[ih].vel[1][il], pred[ih].vel[1][il]},
				    {pred[ih].vel[2][il], pred[ih].vel[2][il]},
				};
				v2df acci[3] = {
				    {pred[ih].acc[0][il], pred[ih].acc[0][il]},
				    {pred[ih].acc[1][il], pred[ih].acc[1][il]},
				    {pred[ih].acc[2][il], pred[ih].acc[2][il]},
				};
				v2df eps2i = {pred[i].eps2[ih], pred[i].eps2[il]};
				v2df idi   = {pred[ih].id[il], pred[ih].id[il]};
				*/
				v2df posi[3] = {
				    {pred[i].pos[0], pred[i].pos[0]},
				    {pred[i].pos[1], pred[i].pos[1]},
				    {pred[i].pos[2], pred[i].pos[2]},
				};
				//cout << i <<"  "<< pred[i].pos[0];
				v2df veli[3] = {
				    {pred[i].vel[0], pred[i].vel[0]},
				    {pred[i].vel[1], pred[i].vel[1]},
				    {pred[i].vel[2], pred[i].vel[2]},
				};
				//cout <<"  "<< pred[i].vel[0];
				v2df acci[3] = {
				    {pred[i].acc[0], pred[i].acc[0]},
				    {pred[i].acc[1], pred[i].acc[1]},
				    {pred[i].acc[2], pred[i].acc[2]},
				};
				//cout <<"  "<< pred[i].acc[0] << endl;
				v2df eps2i = {eps2[i], eps2[i]};
				v2df idi   = {pred[i].id, pred[i].id};
				calc_force_on_i(force[i], posi, veli, acci,
                                                eps2i, idi, js, je);
                            } // end parallel for(i)
                        }
		    }
	    void calc_force_on_predictors(
	    int ni,
	    std::vector<predictor> &pred,
	    std::vector<Force> &force,
	    int nc)
	  {
	      //cerr << "nc=" << nc << endl;
	    for(unsigned jb=0; jb<(unsigned)njblocks; jb++){
		unsigned nbody2 = n / 2;
		unsigned js = ((jb  )*nbody2) / njblocks;
		unsigned je = ((jb+1)*nbody2) / njblocks;
		//cerr << "js=" << js << " je=" << je << endl;
		if(je>nbody2) je = nbody2;
#pragma omp parallel for
		for(int i=0; i<ni; i++){
		    v2df posi[3] = {
			{pred[i].pos[0], pred[i].pos[0]},
			{pred[i].pos[1], pred[i].pos[1]},
			{pred[i].pos[2], pred[i].pos[2]},
		    };
		    //cout << i <<"  "<< pred[i].pos[0];
		    v2df veli[3] = {
			{pred[i].vel[0], pred[i].vel[0]},
			{pred[i].vel[1], pred[i].vel[1]},
			{pred[i].vel[2], pred[i].vel[2]},
		    };
		    //cout <<"  "<< pred[i].vel[0];
		    v2df acci[3] = {
			{pred[i].acc[0], pred[i].acc[0]},
			{pred[i].acc[1], pred[i].acc[1]},
			{pred[i].acc[2], pred[i].acc[2]},
		    };
		    //cout <<"  "<< pred[i].acc[0] << endl;
		    //v2df eps2i = {eps2[i], eps2[i]};
		    v2df eps2i = {pred[i].eps2, pred[i].eps2};
		    double id = pred[i].id;
		    v2df idi   = {id, id};

		    if((int)id<nc){
			//// for cluster particle ////
			calc_force_on_i_epsj(force[i], posi, veli, acci,
					     idi, js, je);
			//cerr << id << endl;
		    }else{
			//cerr << "test" << endl;
			//// for tree & BH particles ////
			calc_force_on_i(force[i], posi, veli, acci,
					eps2i, idi, js, je);
		    }
		    //cout << force[i].acc[0] << endl;
		} // end parallel for(i)
	    }
	}
	void calc_force_on_predictors(
				      int ni,
				      predictor pred[],
				      Force force[],
				      int nj)
	  {
	    if(nj%2 != 0) nj++;
	    for(unsigned jb=0; jb<(unsigned)njblocks; jb++){
	      //unsigned nbody2 = n / 2;
	        unsigned nbody2 = nj / 2;
		unsigned js = ((jb  )*nbody2) / njblocks;
		unsigned je = ((jb+1)*nbody2) / njblocks;

		if(je>nbody2) je = nbody2;
#pragma omp parallel for
		for(int i=0; i<ni; i++){
		    v2df posi[3] = {
			{pred[i].pos[0], pred[i].pos[0]},
			{pred[i].pos[1], pred[i].pos[1]},
			{pred[i].pos[2], pred[i].pos[2]},
		    };
		    //cout << i <<"  "<< pred[i].pos[0];
		    v2df veli[3] = {
			{pred[i].vel[0], pred[i].vel[0]},
			{pred[i].vel[1], pred[i].vel[1]},
			{pred[i].vel[2], pred[i].vel[2]},
		    };
		    //cout <<"  "<< pred[i].vel[0];
		    v2df acci[3] = {
			{pred[i].acc[0], pred[i].acc[0]},
			{pred[i].acc[1], pred[i].acc[1]},
			{pred[i].acc[2], pred[i].acc[2]},
		    };
		    //cout <<"  "<< pred[i].acc[0] << endl;
		    v2df eps2i = {pred[i].eps2, pred[i].eps2};
		    double id = pred[i].id;
		    v2df idi   = {id, id};
		    calc_force_on_i(force[i], posi, veli, acci,
				    eps2i, idi, js, je);
		} // end parallel for(i)
	    }
	}
	void calc_force_on_predictors_epsj(
					   int ni,
					   predictor pred[],
					   Force force[],
					   int nj)
	  {
	    if(nj%2 != 0) nj++;
	    for(unsigned jb=0; jb<(unsigned)njblocks; jb++){
	      //unsigned nbody2 = n / 2;
	        unsigned nbody2 = nj / 2;
		unsigned js = ((jb  )*nbody2) / njblocks;
		unsigned je = ((jb+1)*nbody2) / njblocks;

		if(je>nbody2) je = nbody2;
#pragma omp parallel for
		for(int i=0; i<ni; i++){
		    v2df posi[3] = {
			{pred[i].pos[0], pred[i].pos[0]},
			{pred[i].pos[1], pred[i].pos[1]},
			{pred[i].pos[2], pred[i].pos[2]},
		    };
		    //cout << i <<"  "<< pred[i].pos[0];
		    v2df veli[3] = {
			{pred[i].vel[0], pred[i].vel[0]},
			{pred[i].vel[1], pred[i].vel[1]},
			{pred[i].vel[2], pred[i].vel[2]},
		    };
		    //cout <<"  "<< pred[i].vel[0];
		    v2df acci[3] = {
			{pred[i].acc[0], pred[i].acc[0]},
			{pred[i].acc[1], pred[i].acc[1]},
			{pred[i].acc[2], pred[i].acc[2]},
		    };
		    //cout <<"  "<< pred[i].acc[0] << endl;
		    //v2df eps2i = {pred[i].eps2, pred[i].eps2};
		    double id = pred[i].id;
		    v2df idi   = {id, id};
		    calc_force_on_i_epsj(force[i], posi, veli, acci,
					 idi, js, je);
		} // end parallel for(i)
	    }
	}
void calc_force_on_predictors(
	    int ni,
	    std::vector<predictor> &pred,
	    std::vector<Force> &force,
	    //std::vector<float> &eps2,
	    int nc,
	    int j0,
	    int j1)
	  {
	      //cerr << "nc=" << nc << endl;	      
	      unsigned nj = j1 - j0;
	      //cerr << "nj=" << nj << endl;
	      njblocks = nj / JBSIZE + (nj % JBSIZE ? 1 : 0);
	    for(unsigned jb=0; jb<(unsigned)njblocks; jb++){
		unsigned nbody2 = nj / 2;//n / 2;
		unsigned js = ((jb  )*nbody2) / njblocks + j0/2;
		unsigned je = ((jb+1)*nbody2) / njblocks + j0/2;		
		if((int)je>j1/2) je = j1/2;
		//cout << "js=" << js << " je=" << je << endl;
#pragma omp parallel for
		for(int i=0; i<ni; i++){
		    v2df posi[3] = {
			{pred[i].pos[0], pred[i].pos[0]},
			{pred[i].pos[1], pred[i].pos[1]},
			{pred[i].pos[2], pred[i].pos[2]},
		    };
		    //cout << i <<"  "<< pred[i].pos[0];
		    v2df veli[3] = {
			{pred[i].vel[0], pred[i].vel[0]},
			{pred[i].vel[1], pred[i].vel[1]},
			{pred[i].vel[2], pred[i].vel[2]},
		    };
		    //cout <<"  "<< pred[i].vel[0];
		    v2df acci[3] = {
			{pred[i].acc[0], pred[i].acc[0]},
			{pred[i].acc[1], pred[i].acc[1]},
			{pred[i].acc[2], pred[i].acc[2]},
		    };
		    //cout <<"  "<< pred[i].acc[0] << endl;
		    //v2df eps2i = {eps2[i], eps2[i]};
		    v2df eps2i = {pred[i].eps2, pred[i].eps2};
		    double id = pred[i].id;
		    v2df idi   = {id, id};

		    if((int)id<nc){
			//// for cluster particle ////
			calc_force_on_i_epsj(force[i], posi, veli, acci,
					     idi, js, je);
/*
			cout << i << "  " << pred[i].eps2  << endl;
			cout << i << "  " << pred[i].pos[0] << "  " << pred[i].pos[1] << "  " << pred[i].pos[2]  << endl;
			cout << i << "  " << pred[i].vel[0] << "  " << pred[i].vel[1] << "  " << pred[i].vel[2]  << endl;
			cout << i << "  " << pred[i].acc[0] << "  " << pred[i].acc[1] << "  " << pred[i].acc[2] << endl;
			cout << i << "  " << force[i].acc[0] << "  " << force[i].acc[1] << "  " << force[i].acc[2] << endl;
*/
			//cerr << id << endl;
		    }else{
			//cerr << "test" << endl;
			//// for tree & BH particles ////
			calc_force_on_i(force[i], posi, veli, acci,
					eps2i, idi, js, je);
		    }
		    //cout << i << "  "<< force[i].phi << endl;
		} // end parallel for(i)
	    }
	}		

/*
		void calculate_force_on_first_ni_particles(
				unsigned ni,
				float eps2)
		{
			for(int i=0; i<(int)ni; i++){
				fobuf[i].clear();
			}
			for(unsigned jb=0; jb<njblocks; jb++){
				unsigned nbody2 = nbody / 2;
				unsigned js = ((jb  )*nbody2) / njblocks;
				unsigned je = ((jb+1)*nbody2) / njblocks;
#pragma omp parallel for
				for(int i=0; i<(int)ni; i++){
					// const predictor &pi = pred[i];
					unsigned ih = i/2;
					unsigned il = i%2;
					v2df posi[3] = {
						{pred[ih].pos[0][il], pred[ih].pos[0][il]},
						{pred[ih].pos[1][il], pred[ih].pos[1][il]},
						{pred[ih].pos[2][il], pred[ih].pos[2][il]},
					};
					v2df veli[3] = {
						{pred[ih].vel[0][il], pred[ih].vel[0][il]},
						{pred[ih].vel[1][il], pred[ih].vel[1][il]},
						{pred[ih].vel[2][il], pred[ih].vel[2][il]},
					};
					v2df acci[3] = {
						{pred[ih].acc[0][il], pred[ih].acc[0][il]},
						{pred[ih].acc[1][il], pred[ih].acc[1][il]},
						{pred[ih].acc[2][il], pred[ih].acc[2][il]},
					};
					v2df eps2i = {eps2, eps2};
					v2df idi   = {pred[ih].id[il], pred[ih].id[il]};
					calc_force_on_i(i, posi, veli, acci, eps2i, idi, js, je);
				} // end parallel for(i)
			}
		}
*/
		void calc_pot_on_predictors(int ni,
					    std::vector<predictor> &pred,
					    std::vector<Force> &force,
					    std::vector<float> &eps2){}

		void calculate_potential(float ) {}
		void debug(unsigned i){
			unsigned ih = i/2;
			unsigned il = i%2;
			std::cerr << ptcl[ih].vel[0][il] << " "
			          << ptcl[ih].vel[1][il] << " "
			          << ptcl[ih].vel[2][il] << std::endl;
			std::cerr << pred2[ih].vel[0][il] << " "
			          << pred2[ih].vel[1][il] << " "
			          << pred2[ih].vel[2][il] << std::endl;
		}
	};
}

#endif
