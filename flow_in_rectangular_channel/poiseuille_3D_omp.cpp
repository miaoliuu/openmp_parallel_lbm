/** 该程序用于计算恒定速度入口流
 *  
 *  注意 NEEM 方法一定要先计算边，然后计算角点
 *  这样做是为了确保计算角点时不会取到边上未更新的分布函数值
 * 
 *  (xy plane): z = 0  恒速度入口，z == nz-1 恒压力出口且流向速度充分发展切向速度为 0 
 * 
 *  角点作为出入口处理
 */

// open Debug
// #define MY_DEBUG

// using Incompressible SRT Equillibrium
#define MY_INCOMPRESSIBLE

#define MY_OMP_OPEN
#ifdef MY_OMP_OPEN
    #include <omp.h>
#endif

#include "src/GlobalDef.hpp"
#include "src/Array.hpp"
#include "src/Field.hpp"
#include "src/Units.hpp"
#include "src/D3Q19BGK.hpp"
#include <ctime>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>

typedef Real T;
typedef units::IncompFlowParam<T> SimParam;
typedef lbmodels::d3q19::D3Q19BGK<T> LBM;

LBM::DensityField rhoF;
LBM::VelocityField velF, velF0;
LBM::LatticeField latticeF, latticeF0;

T feq_i(int i, LBM::Density_t rho, LBM::Velocity_t vel, T uSqr);
void init(SimParam const& param);
void evolution(SimParam const& param);
T ErrorVel();
T AverageRho();
void writeData(int t);

int main(int argc, char const *argv[])
{
#ifdef MY_OMP_OPEN
    std::cout << "Number of processer used: " << omp_get_num_procs() << std::endl;
#endif
    time_t t_start, t_end;
    T avg_rho, err_vel;
    int info_step = 100;
    int out_step = 2000;

    T Re = 10.0;
    T la_Umax = 0.05; // inlet velocity
    T la_Cs = LBM::cs;
    int resolution = 100;
    T lx = 1;
    T ly = 1;
    T lz = 5;
    SimParam const ldc_param(Re, la_Umax, la_Cs, resolution, lx, ly, lz);
    std::cout << ldc_param;

    init(ldc_param);
    writeData(0);
    t_start = time(NULL);
    for (int t=0; true; ++t) {
        evolution(ldc_param);
        if (t%info_step == 0) {
            avg_rho = AverageRho();
            err_vel = ErrorVel();
            t_end = time(NULL);
            std::cout << "[" << difftime(t_end, t_start) << " s]"
                      << " Time step: " << t
                      << " avg_rho: " << std::setprecision(6) << avg_rho
                      << " relative error: " 
                      << std::setiosflags(std::ios::scientific)
                      << err_vel
                      << std::resetiosflags(std::ios::scientific)
                      << std::endl;
            if ( t >= out_step) {
                if (t%out_step == 0) { writeData(t); }
                if ( err_vel < 1.0e-10 ) {
                    writeData(t);
                    break;
                }
            }
        }
    }
    return 0;
}

T feq_i(int i, LBM::Density_t rho, LBM::Velocity_t u, T uSqr)
{
    T feq, uci;
    uci = LBM::c[i][0]*u[0] + LBM::c[i][1]*u[1] + LBM::c[i][2]*u[2];
#ifdef MY_INCOMPRESSIBLE
    feq = LBM::t[i]*(rho + LBM::invCs2*uci + 0.5*LBM::invCs2*(LBM::invCs2*uci*uci - uSqr));
#else
    feq = rho*LBM::t[i]*(1.0 + LBM::invCs2*uci + 0.5*LBM::invCs2*(LBM::invCs2*uci*uci - uSqr));
#endif
    return feq;
}

void init(SimParam const& param)
{
    T rho0 = 1.0;
    T U0 = param.GetLatticeU();
    T nx = param.GetNx();
    T ny = param.GetNy();
    T nz = param.GetNz();
    rhoF = LBM::DensityField(nx,ny,nz);
    velF = LBM::VelocityField(nx,ny,nz);
    velF0 = LBM::VelocityField(nx,ny,nz);
    latticeF = LBM::LatticeField(nx,ny,nz);
    latticeF0 = LBM::LatticeField(nx,ny,nz);

    for (int z=0; z<nz; ++z) {
        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                rhoF(x,y,z) = rho0;
                velF(x,y,z)[0] = 0.0;
                velF(x,y,z)[1] = 0.0;
                velF(x,y,z)[2] = 0.0;
                velF(x,y,0)[2] = U0; // z == 0 处为速度入口 vel-z == U0
                T uSqr = pow(velF(x,y,z)[0],2) + pow(velF(x,y,z)[1],2) + pow(velF(x,y,z)[2],2);
                for (int i=0; i<LBM::q; ++i) {
                    latticeF(x,y,z)[i] = feq_i(i, rhoF(x,y,z), velF(x,y,z), uSqr);
                }
            }
        }
    }
}

void evolution(SimParam const& param)
{
    int const nx = param.GetNx();
    int const ny = param.GetNy();
    int const nz = param.GetNz();
    T const omega = param.GetOmega();
    // Bulk
    // ---- collision and streaming (separate) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int z=0; z<nz; ++z) {
        for (int y=0; y<ny; ++y) {
            for (int x=0; x<nx; ++x) {
                T uSqr = pow(velF(x,y,z)[0],2) + pow(velF(x,y,z)[1],2) + pow(velF(x,y,z)[2],2);
                for (int i=0; i<LBM::q; ++i) {
                    latticeF(x,y,z)[i] *= (1.0-omega);
                    latticeF(x,y,z)[i] += omega*feq_i(i, rhoF(x,y,z), velF(x,y,z), uSqr);
                }
            }
        }
    }
    // ---- streaming just bulk (boundary not streaming) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int z=1; z<nz-1; ++z) {
        for (int y=1; y<ny-1; ++y) {
            for (int x=1; x<nx-1; ++x) {
                for (int i=0; i<LBM::q; ++i) {
                    int ix = x - LBM::c[i][0];
                    int iy = y - LBM::c[i][1];
                    int iz = z - LBM::c[i][2];
                    latticeF0(x,y,z)[i] = latticeF(ix,iy,iz)[i];
                }
            }
        }
    }
    // ---- update Micro of bulk ----
    latticeF.Swap(latticeF0);
    // ---- Update Macro of bulk ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int z=1; z<nz-1; ++z) {
        for (int y=1; y<ny-1; ++y) {
            for (int x=1; x<nx-1; ++x) {
                velF0(x,y,z)[0] = velF(x,y,z)[0];
                velF0(x,y,z)[1] = velF(x,y,z)[1];
                velF0(x,y,z)[2] = velF(x,y,z)[2];
                T m0 = 0.0;    // 0-order momentum
                T m1_x = 0.0;  // 1-order momentum at x-direction
                T m1_y = 0.0;  // 1-order momentum at y-direction
                T m1_z = 0.0;  // 1-order momentum at z-direction
                for (int i=0; i<LBM::q; ++i) {
                    m0 += latticeF(x,y,z)[i];
                    m1_x += latticeF(x,y,z)[i] * LBM::c[i][0];
                    m1_y += latticeF(x,y,z)[i] * LBM::c[i][1];
                    m1_z += latticeF(x,y,z)[i] * LBM::c[i][2];
                }
                rhoF(x,y,z) = m0; // Update Density
            #ifdef MY_INCOMPRESSIBLE
                velF(x,y,z)[0] = m1_x;
                velF(x,y,z)[1] = m1_y;
                velF(x,y,z)[2] = m1_z;
            #else
                velF(x,y,z)[0] = m1_x / m0;
                velF(x,y,z)[1] = m1_y / m0;
                velF(x,y,z)[2] = m1_z / m0;
            #endif
            }
        }
    }
    // boundary (NEEM)
    // ---- X-direction (YZ-planes without edges) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif 
    for (int z=1; z<nz-1; ++z) {
        for (int y=1; y<ny-1; ++y) {
            T uSqr_1; // for boundary
            T uSqr_2; // for bulk
            // ---- left (x = 0, wall) ----
            rhoF(0,y,z) = rhoF(1,y,z);
            uSqr_1 = pow(velF(0,y,z)[0],2) + pow(velF(0,y,z)[1],2) + pow(velF(0,y,z)[2],2);
            uSqr_2 = pow(velF(1,y,z)[0],2) + pow(velF(1,y,z)[1],2) + pow(velF(1,y,z)[2],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(0,y,z)[i] = feq_i(i, rhoF(0,y,z), velF(0,y,z), uSqr_1);
                latticeF(0,y,z)[i] += latticeF(1,y,z)[i] - feq_i(i, rhoF(1,y,z), velF(1,y,z), uSqr_2);
            }
            // ---- left (x = nx-1, wall) ----
            rhoF(nx-1,y,z) = rhoF(nx-2,y,z);
            uSqr_1 = pow(velF(nx-1,y,z)[0],2) + pow(velF(nx-1,y,z)[1],2) + pow(velF(nx-1,y,z)[2],2);
            uSqr_2 = pow(velF(nx-2,y,z)[0],2) + pow(velF(nx-2,y,z)[1],2) + pow(velF(nx-2,y,z)[2],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(nx-1,y,z)[i] = feq_i(i, rhoF(nx-1,y,z), velF(nx-1,y,z), uSqr_1);
                latticeF(nx-1,y,z)[i] += latticeF(nx-2,y,z)[i] - feq_i(i, rhoF(nx-2,y,z), velF(nx-2,y,z), uSqr_2);
            }
        }
    }
    // ---- Y-direction (XZ-planes with Z-direction edges. Means x=[0,nx) ) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif 
    for (int z=1; z<nz-1; ++z) {
        for (int x=0; x<nx; ++x) {
            T uSqr_1; // for boundary
            T uSqr_2; // for bulk
            // ---- bottom (y = 0, wall) ----
            rhoF(x,0,z) = rhoF(x,1,z);
            uSqr_1 = pow(velF(x,0,z)[0],2) + pow(velF(x,0,z)[1],2) + pow(velF(x,0,z)[2],2);
            uSqr_2 = pow(velF(x,1,z)[0],2) + pow(velF(x,1,z)[1],2) + pow(velF(x,1,z)[2],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,0,z)[i] = feq_i(i, rhoF(x,0,z), velF(x,0,z), uSqr_1);
                latticeF(x,0,z)[i] += latticeF(x,1,z)[i] - feq_i(i, rhoF(x,1,z), velF(x,1,z), uSqr_2);
            }
            // ---- top (y = ny-1, wall) ----
            rhoF(x,ny-1,z) = rhoF(x,ny-2,z);
            uSqr_1 = pow(velF(x,ny-1,z)[0],2) + pow(velF(x,ny-1,z)[1],2) + pow(velF(x,ny-1,z)[2],2);
            uSqr_2 = pow(velF(x,ny-2,z)[0],2) + pow(velF(x,ny-2,z)[1],2) + pow(velF(x,ny-2,z)[2],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,ny-1,z)[i] = feq_i(i, rhoF(x,ny-1,z), velF(x,ny-1,z), uSqr_1);
                latticeF(x,ny-1,z)[i] += latticeF(x,ny-2,z)[i] - feq_i(i, rhoF(x,ny-2,z), velF(x,ny-2,z), uSqr_2);
            }
        }
    }
    // ---- Z-direction (XY-planes with edges and corners. Means x=[0,nx), y=[0,ny) ) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif 
    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            T uSqr_1; // for boundary
            T uSqr_2; // for bulk
            // ---- inlet (z = 0) ----
            rhoF(x,y,0) = rhoF(x,y,1);
            uSqr_1 = pow(velF(x,y,0)[0],2) + pow(velF(x,y,0)[1],2) + pow(velF(x,y,0)[2],2);
            uSqr_2 = pow(velF(x,y,1)[0],2) + pow(velF(x,y,1)[1],2) + pow(velF(x,y,1)[2],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,y,0)[i] = feq_i(i, rhoF(x,y,0), velF(x,y,0), uSqr_1);
                latticeF(x,y,0)[i] += latticeF(x,y,1)[i] - feq_i(i, rhoF(x,y,1), velF(x,y,1), uSqr_2);
            }
            // ---- outlet (z = nz-1) ----
            // rhoF(x,y,nz-1) = rhoF(x,y,nz-2);
            rhoF(x,y,nz-1) = 1.0;
            velF(x,y,nz-1)[2] = velF(x,y,nz-2)[2]; // Vel-Z fully developed
            uSqr_1 = pow(velF(x,y,nz-1)[0],2) + pow(velF(x,y,nz-1)[1],2) + pow(velF(x,y,nz-1)[2],2);
            uSqr_2 = pow(velF(x,y,nz-2)[0],2) + pow(velF(x,y,nz-2)[1],2) + pow(velF(x,y,nz-2)[2],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,y,nz-1)[i] = feq_i(i, rhoF(x,y,nz-1), velF(x,y,nz-1), uSqr_1);
                latticeF(x,y,nz-1)[i] += latticeF(x,y,nz-2)[i] - feq_i(i, rhoF(x,y,nz-2), velF(x,y,nz-2), uSqr_2);
            }
        }
    }
}

T ErrorVel()
{
    int nx = velF.GetNX();
    int ny = velF.GetNY();
    int nz = velF.GetNZ();
    T result = 0.0;
    T temp1 = 0.0;
    T temp2 = 0.0;
    // Error at bulk
#ifdef MY_OMP_OPEN
    #pragma omp parallel for reduction(+ : temp1, temp2)
#endif
    for (int z=1; z<nz-1; ++z) {
        for (int y=1; y<ny-1; ++y) {
            for (int x=1; x<nx-1; ++x) {
                temp1 += pow(velF(x,y,z)[0]-velF0(x,y,z)[0], 2) + pow(velF(x,y,z)[1]-velF0(x,y,z)[1], 2) + pow(velF(x,y,z)[2]-velF0(x,y,z)[2], 2);
                temp2 += pow(velF(x,y,z)[0], 2) + pow(velF(x,y,z)[1], 2) + pow(velF(x,y,z)[2], 2);
            }
        }
    }
    result = sqrt( temp1/(temp2+1.0e-32) );
    return result;
}

T AverageRho()
{
    int nx = rhoF.GetNX();
    int ny = rhoF.GetNY();
    int nz = velF.GetNZ();
    T result = 0.0;
    // average rho at bulk
#ifdef MY_OMP_OPEN
    #pragma omp parallel for reduction(+ : result)
#endif
    for (int z=1; z<nz-1; ++z) {
        for (int y=1; y<ny-1; ++y) {
            for (int x=1; x<nx-1; ++x) {
                result += rhoF(x,y,z);
            }
        }
    }
    result /= (double)(nx-2)*(double)(ny-2)*(double)(nz-2);
    return result;
}

void writeData(int t)
{
    // vti format
    std::ostringstream file_name;
    file_name << "poiseuille_3D_" << t << ".vti";
    std::ofstream out_file(file_name.str(), std::ios::out | std::ios::trunc);
    if ( !out_file.is_open() ) {
        std::cerr << "Can't open file \"" << file_name.str() << "\"" << std::endl;
        exit(-1);
    }
    out_file << "<?xml version=\"1.0\"?>\n"
             << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out_file << "<ImageData WholeExtent="
             << "\"0 " << rhoF.GetNX()-1 << " 0 " << rhoF.GetNY()-1 << " 0 " << rhoF.GetNZ()-1 << "\" "
             << "Origin=\"0 0 0\" Spacing=\"1 1 1\" Direction=\"1 0 0 0 1 0 0 0 1\">\n";
    out_file << "<Piece Extent=\"0 " << rhoF.GetNX()-1 << " 0 " << rhoF.GetNY()-1 << " 0 " << rhoF.GetNZ()-1 << "\">\n";
    out_file << "<PointData>\n";

    out_file << "<DataArray type=\"Float64\" Name=\"rho\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int k=0; k<rhoF.GetNZ(); ++k) {
        for (int j=0; j<rhoF.GetNY(); ++j) {
            for (int i=0; i<rhoF.GetNX(); ++i) {
                out_file << rhoF(i,j,k) << "\n";
            }
        }
    }
    out_file << "</DataArray>" << std::endl;

    out_file << "<DataArray type=\"Float64\" Name=\"vel\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int k=0; k<rhoF.GetNZ(); ++k) {
        for (int j=0; j<rhoF.GetNY(); ++j) {
            for (int i=0; i<rhoF.GetNX(); ++i) {
                out_file << velF(i,j,k) << "\n";
            }
        }
    }
    out_file << "</DataArray>" << std::endl;

    out_file << "</PointData>\n";
    out_file << "</Piece>\n";
    out_file << "</ImageData>\n";
    out_file << "</VTKFile>" << std::endl;
}
