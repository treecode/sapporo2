/* GRAPE6 API as implemented by Sapporo2
 *
 * See https://www.cfca.nao.ac.jp/files/grape6user.pdf for the documentation.
 *
 * Note that some functions appear to be missing, e.g. g6_set_nip() and
 * g6_set_i_particle_scales_from_real_value().
*/

// Fortran ABI
int g6_open_(int *id);

int g6_close_(int *id);

int g6_npipes_();

int g6_set_tunit_(double*);

int g6_set_xunit_(double*);

int g6_set_ti_(int *id, double *ti);

int g6_set_j_particle_(int *cluster_id,
                       int *address,
                       int *index,
                       double *tj, double *dtj,
                       double *mass,
                       double k18[3], double j6[3],
                       double a2[3], double v[3], double x[3]);

void g6calc_firsthalf_(int *cluster_id,
                       int *nj, int *ni,
                       int index[],
                       double xi[][3], double vi[][3],
                       double aold[][3], double j6old[][3],
                       double phiold[3],
                       double *eps2, double h2[]);

int g6calc_lasthalf_(int *cluster_id,
                     int *nj, int *ni,
                     int index[],
                     double xi[][3], double vi[][3],
                     double *eps2, double h2[],
                     double acc[][3], double jerk[][3], double pot[]);

int g6calc_lasthalf2_(int *cluster_id,
                      int *nj, int *ni,
                      int index[],
                      double xi[][3], double vi[][3],
                      double *eps2, double h2[],
                      double acc[][3], double jerk[][3], double pot[],
                      int *inn);

int g6_initialize_jp_buffer_(int* cluster_id, int* buf_size) { cluster_id = cluster_id; buf_size=buf_size; return 0;}
int g6_flush_jp_buffer_(int* cluster_id) { cluster_id = cluster_id; return 0;}
int g6_reset_(int* cluster_id) {cluster_id = cluster_id; return 0;}
int g6_reset_fofpga_(int* cluster_id) {cluster_id = cluster_id; return 0;}

int g6_read_neighbour_list_(int* cluster_id);

int g6_get_neighbour_list_(int *cluster_id,
                           int *ipipe,
                           int *maxlength,
                           int *n_neighbours,
                           int neighbour_list[]);

// This is not part of the GRAPE6 API, but is useful for debugging.
void get_j_part_data(int addr, int nj,
                     double *pos,
                     double *vel,
                     double *acc,
                     double *jrk,
                     double *ppos,
                     double *pvel,
                     double &mass,
                     double &eps2,
                     int    &id);


// C ABI
// These forward to the Fortran versions above, which are actually implemented by
// Sapporo2.

extern "C" {

inline int g6_open(int id) {
    g6_open_(&id);
}

inline int g6_close(int id) {
    g6_close(&id);
}

inline int g6_npipes() {
    return g6_npipes_();
}

inline int g6_set_tunit(double tu) {
    return g6_set_tunit_(&tu);
}

inline int g6_set_xunit(double xu) {
    return g6_set_xunit_(&xu);
}

inline int g6_set_ti(int id, double ti) {
    return g6_set_ti_(&id, &ti);
}

inline int g6_set_j_particle(int cluster_id,
                             int address,
                             int index,
                             double tj, double dtj,
                             double mass,
                             double k18[3], double j6[3],
                             double a2[3], double v[3], double x[3])
{
    return g6_set_j_particle(
            &cluster_id, &address, &index, &tj, &dtj, &mass, k18, j6, a2, v, x);
}

inline void g6calc_firsthalf(int cluster_id,
                             int nj, int ni,
                             int index[],
                             double xi[][3], double vi[][3],
                             double aold[][3], double j6old[][3],
                             double phiold[3],
                             double eps2, double h2[])
{
    g6calc_firsthalf(
            &cluster_id, &nj, &ni, index, xi, vi, aold, j6old, piold, &eps2, h2);
}

inline int g6calc_lasthalf(int cluster_id,
                           int nj, int ni,
                           int index[],
                           double xi[][3], double vi[][3],
                           double eps2, double h2[],
                           double acc[][3], double jerk[][3], double pot[])
{
   return g6calc_lasthalf_(
           &cluster_id, &nj, &ni, index, xi, vi, &eps2, h2, acc, jerk, pot);
}

inline int g6calc_lasthalf2(int cluster_id,
                            int nj, int ni,
                            int index[],
                            double xi[][3], double vi[][3],
                            double eps2, double h2[],
                            double acc[][3], double jerk[][3], double pot[],
                            int *inn)
{
    return g6calc_lasthalf2_(
            &cluster_id, &nj, &ni, index, xy, vi, &eps2, h2, acc, jerk, pot, inn);
}

inline int g6_initialize_jp_buffer(int cluster_id, int buf_size) {
    return g6_initialize_jp_buffer_(&cluster_id, &buf_size);
}

inline int g6_flush_jp_buffer(int cluster_id) {
    return g6_flush_jp_buffer_(&cluster_id);
}

inline int g6_reset(int cluster_id) {
    return g6_reset_(&cluster_id);
}

inline int g6_reset_fofpga(int cluster_id) {
    return g6_reset_fofpga_(&cluster_id);
}

inline int g6_read_neighbour_list(int cluster_id) {
    return g6_read_neighbour_list_(&cluster_id);
}

inline int g6_get_neighbour_list(int cluster_id,
                                 int ipipe,
                                 int maxlength,
                                 int n_neighbours,
                                 int neighbour_list[])
{
    return g6_get_neighbour_list(
            &cluster_id, &ipipe, &maxlength, &n_neighbours, neighbour_list);
}

}

