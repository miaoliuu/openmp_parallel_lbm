/**
 *  注意 NEEM 方法一定要先计算边，然后计算角点
 *  这样做是为了确保计算角点时不会取到边上未更新的分布函数值
 * 
 *  分离式 SWAP 算法只需要保证 c[i][] = -c[i+half][] 即可
 * 
 *  Combined SWAP 算法还要考虑 c 的方向，坐标循环的方向，同时 boundary 与 bulk 还需分离计算
 */

// open Debug
// #define MY_DEBUG

#define MY_OMP_OPEN
#ifdef MY_OMP_OPEN
    #include <omp.h>
#endif

#include "src/GlobalDef.hpp"
#include "src/Array.hpp"
#include "src/Field.hpp"
#include "src/Units.hpp"
#include "src/D2Q9BGK.hpp"
#include <time.h>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>

typedef Real T;
typedef units::IncompFlowParam<T> SimParam;
typedef lbmodels::d2q9::D2Q9BGK<T> LBM;

LBM::DensityField rhoF;
LBM::VelocityField velF, velF0;
LBM::LatticeField latticeF;

bool const swap_combine = false;

void init(SimParam const& param);
T feq_i(int i, LBM::Density_t rho, LBM::Velocity_t vel);
void evolution(SimParam const& param);
T ErrorVel();
T AverageRho();
void writeData(int t);
void write_stream_func(int t);

int main(int argc, char const *argv[])
{
    time_t t_start, t_end;
    T avg_rho, err_vel;
    int info_step = 100;
    int out_step = 2000;

    T Re = 1000.0;
    T la_Umax = 0.1; // top lid velocity
    T la_Cs = LBM::cs;
    int resolution = 256;
    T lx = 1;
    T ly = 1;
    SimParam const ldc_param(Re, la_Umax, la_Cs, resolution, lx, ly);
    // std::cout << ldc_param;

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
                if ( err_vel < 1.0e-6 ) {
                    write_stream_func(t);
                    break;
                }
            }
        }
    }
    return 0;
}

void init(SimParam const& param)
{
    T rho0 = 1.0;
    T U0 = param.GetLatticeU();
    T nx = param.GetNx();
    T ny = param.GetNy();
    rhoF = LBM::DensityField(nx,ny);
    velF = LBM::VelocityField(nx,ny);
    velF0 = LBM::VelocityField(nx,ny);
    latticeF = LBM::LatticeField(nx,ny);

    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            rhoF(x,y) = rho0;
            velF(x,y)[0] = 0.0;
            velF(x,y)[1] = 0.0;
            velF(x, ny-1)[0] = U0;
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,y)[i] = feq_i(i, rhoF(x,y), velF(x,y));
            }
        }
    }
}

T feq_i(int i, LBM::Density_t rho, LBM::Velocity_t u)
{
    T feq;
    T uu = u[0]*u[0] + u[1]*u[1];
    T uci = LBM::c[i][0]*u[0] + LBM::c[i][1]*u[1];
    feq = rho*LBM::t[i]*(1.0 + LBM::invCs2*uci + 0.5*LBM::invCs2*(LBM::invCs2*uci*uci - uu));
    return feq;
}

void evolution(SimParam const& param)
{
    int const nx = param.GetNx();
    int const ny = param.GetNy();
    T const omega = param.GetOmega();
if ( swap_combine )
{
    // ---- collision and streaming (combined-SWAP algorithm) ----
    int const half = (LBM::q-1)/2;
    // ---- 1st. Boundary collision swap ----
    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            // lattice on boundary
            if ( x==0 || x==nx-1 || y==0 || y==ny-1 ) {
                // boundary collision
                for (int i=0; i<LBM::q; ++i) {
                    latticeF(x,y)[i] *= (1.0-omega);
                    latticeF(x,y)[i] += omega*feq_i(i, rhoF(x,y), velF(x,y));
                }
                // boundary swap
                for (int i=1; i<=half; ++i) {
                    std::swap(latticeF(x,y)[i], latticeF(x,y)[i+half]);
                }
            }
        }
    }
    // ---- 2nd. Bulk combined swap (collision and streaming) ----
    // 因为 Combined SWAP 受格子方向影响，这里 y 必须优先循环
    for (int x=1; x<nx-1; ++x) {
        for (int y=1; y<ny-1; ++y) {
            // collision
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,y)[i] *= (1.0-omega);
                latticeF(x,y)[i] += omega*feq_i(i, rhoF(x,y), velF(x,y));
            }
            // combined swap, from 1 to half (with half)
            for (int i=1; i<=half; ++i) {
                int ix = x + LBM::c[i][0];
                int iy = y + LBM::c[i][1];
                if ( ix>=0 && ix<nx && iy>=0 && iy<ny ) {
                    // T temp = latticeF(x,y)[i];
                    // latticeF(x,y)[i] = latticeF(x,y)[i+half];
                    // latticeF(x,y)[i+half] = latticeF(ix,iy)[i];
                    // latticeF(ix,iy)[i] = temp;
                    std::swap(latticeF(x,y)[i], latticeF(x,y)[i+half]); // swap collided lattice
                    std::swap(latticeF(x,y)[i+half], latticeF(ix,iy)[i]); // streaming swap
                }
            }
        }
    }
    // ---- 3rd. Boundary streaming swap ----
    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            // lattice on boundary
            if ( x==0 || x==nx-1 || y==0 || y==ny-1 ) {
                // streaming swap on boundary
                for (int i=1; i<=half; ++i) {
                    int ix = x + LBM::c[i][0];
                    int iy = y + LBM::c[i][1];
                    if ( ix>=0 && ix<nx && iy>=0 && iy<ny ) {
                        std::swap(latticeF(x,y)[i+half], latticeF(ix,iy)[i]); // streaming swap
                    }
                }
            }
        }
    }
}
else
{
    // ---- collision and streaming (SWAP algorithm) ----
    int const half = (LBM::q-1)/2;
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,y)[i] *= (1.0-omega);
                latticeF(x,y)[i] += omega*feq_i(i, rhoF(x,y), velF(x,y));
            }
            // from 1 to half (with half)
            for (int i=1; i<=half; ++i) { 
                std::swap(latticeF(x,y)[i], latticeF(x,y)[i+half]);
            }
        }
    }
    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            // from 1 to half (with half)
            for (int i=1; i<=half; ++i) {
                int ix = x + LBM::c[i][0];
                int iy = y + LBM::c[i][1];
                // int ix = x - LBM::c[i][0];
                // int iy = y - LBM::c[i][1];
                if ( ix>=0 && ix<nx && iy>=0 && iy<ny ) {
                    std::swap(latticeF(x,y)[i+half], latticeF(ix,iy)[i]);
                    // std::swap(latticeF(x,y)[i], latticeF(ix,iy)[i+half]);
                }
            }
        }
    }
}
    // ---- Update Macro at bulk ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int y=1; y<ny-1; ++y) {
        for (int x=1; x<nx-1; ++x) {
            velF0(x,y)[0] = velF(x,y)[0];
            velF0(x,y)[1] = velF(x,y)[1];
            T m0 = 0.0;    // 0-order momentum
            T m1_x = 0.0;  // 1-order momentum at x-direction
            T m1_y = 0.0;  // 1-order momentum at y-direction
            for (int i=0; i<LBM::q; ++i) {
                m0 += latticeF(x,y)[i];
                m1_x += latticeF(x,y)[i] * LBM::c[i][0];
                m1_y += latticeF(x,y)[i] * LBM::c[i][1];
            }
            rhoF(x,y) = m0;
            velF(x,y)[0] = m1_x / m0;
            velF(x,y)[1] = m1_y / m0;
        }
    }
    // boundary (NEEM)
    // ---- edges (Y-direction) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int y=1; y<ny-1; ++y) {
        // ---- ----
        rhoF(0,y) = rhoF(1,y);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(0,y)[i] = feq_i(i, rhoF(0,y), velF(0,y));
            latticeF(0,y)[i] += latticeF(1,y)[i] - feq_i(i, rhoF(1,y), velF(1,y));
        }
        // ---- ----
        rhoF(nx-1,y) = rhoF(nx-2,y);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(nx-1,y)[i] = feq_i(i, rhoF(nx-1,y), velF(nx-1,y));
            latticeF(nx-1,y)[i] += latticeF(nx-2,y)[i] - feq_i(i, rhoF(nx-2,y), velF(nx-2,y));
        }
    }
    // ---- edges (X-direction) with corner ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int x=0; x<nx; ++x) {
        // ---- ----
        rhoF(x,0) = rhoF(x,1);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(x,0)[i] = feq_i(i, rhoF(x,0), velF(x,0));
            latticeF(x,0)[i] += latticeF(x,1)[i] - feq_i(i, rhoF(x,1), velF(x,1));
        }
        // ---- ----
        rhoF(x,ny-1) = rhoF(x,ny-2);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(x,ny-1)[i] = feq_i(i, rhoF(x,ny-1), velF(x,ny-1));
            latticeF(x,ny-1)[i] += latticeF(x,ny-2)[i] - feq_i(i, rhoF(x,ny-2), velF(x,ny-2));
        }
    }
}

T ErrorVel()
{
    int nx = velF.GetNX();
    int ny = velF.GetNY();
    T result = 0.0;
    T temp1 = 0.0;
    T temp2 = 0.0;
    // Error at bulk
#ifdef MY_OMP_OPEN
    #pragma omp parallel for reduction(+ : temp1, temp2)
#endif
    for (int y=1; y<ny-1; ++y) {
        for (int x=1; x<nx-1; ++x) {
            temp1 += pow(velF(x,y)[0]-velF0(x,y)[0], 2) + pow(velF(x,y)[1]-velF0(x,y)[1], 2);
            temp2 += pow(velF(x,y)[0], 2) + pow(velF(x,y)[1], 2);
        }
    }
    result = sqrt( temp1/(temp2+1.0e-32) );
    return result;
}

T AverageRho()
{
    int nx = rhoF.GetNX();
    int ny = rhoF.GetNY();
    T result = 0.0;
    // average rho at bulk
#ifdef MY_OMP_OPEN
    #pragma omp parallel for reduction(+ : result)
#endif
    for (int y=1; y<ny-1; ++y) {
        for (int x=1; x<nx-1; ++x) {
            result += rhoF(x,y);
        }
    }
    result /= (double)(nx-2)*(double)(ny-2);
    return result;
}

void writeData(int t)
{
    // vti format
    std::ostringstream file_name;
    file_name << "ldc_" << t << ".vti";
    std::ofstream out_file(file_name.str(), std::ios::out | std::ios::trunc);
    if ( !out_file.is_open() ) {
        std::cerr << "Can't open file \"" << file_name.str() << "\"" << std::endl;
        exit(-1);
    }
    out_file << "<?xml version=\"1.0\"?>\n"
             << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out_file << "<ImageData WholeExtent="
             << "\"0 " << rhoF.GetNX()-1 << " 0 " << rhoF.GetNY()-1 << " 0 0\" "
             << "Origin=\"0 0 0\" Spacing=\"1 1 1\" Direction=\"1 0 0 0 1 0 0 0 1\">\n";
    out_file << "<Piece Extent=\"0 " << rhoF.GetNX()-1 << " 0 " << rhoF.GetNY()-1 << " 0 0\">\n";
    out_file << "<PointData>\n";

    out_file << "<DataArray type=\"Float64\" Name=\"rho\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int j=0; j<rhoF.GetNY(); ++j) {
        for (int i=0; i<rhoF.GetNX(); ++i) {
            out_file << rhoF(i,j) << "\n";
        }
    }
    out_file << "</DataArray>" << std::endl;

    out_file << "<DataArray type=\"Float64\" Name=\"vel\" NumberOfComponents=\"2\" format=\"ascii\">\n";
    for (int j=0; j<velF.GetNY(); ++j) {
        for (int i=0; i<velF.GetNX(); ++i) {
            out_file << velF(i,j) << "\n";
        }
    }
    out_file << "</DataArray>" << std::endl;

    out_file << "</PointData>\n";
    out_file << "</Piece>\n";
    out_file << "</ImageData>\n";
    out_file << "</VTKFile>" << std::endl;
}

void write_stream_func(int t)
{
    // calculate stream-function value
    int nx = rhoF.GetNX();
    int ny = rhoF.GetNY();
    LBM::DensityField strF(nx, ny);
    for (int i=0; i<nx; ++i) {
        if (i != 0) {
            double rho_x = 0.5*(rhoF(i-1,0) + rhoF(i,0));
            double v_x = 0.5*(velF(i-1,0)[1] + velF(i,0)[1]);
            strF(i,0) = strF(i-1,0) - rho_x*v_x;
        }
        for (int j=1; j<ny; ++j) {
            double rho_y = 0.5*(rhoF(i,j-1) + rhoF(i,j));
            double u_y = 0.5*(velF(i,j-1)[0] + velF(i,j)[0]);
            strF(i,j) = strF(i,j-1) + rho_y*u_y;
        }
    }
    // write to vti
    std::ostringstream strF_file_name;
    strF_file_name << "ldc_strF_" << t << ".vti";
    std::ofstream out_file(strF_file_name.str().c_str());
    if ( !out_file.is_open() ) {
        std::cerr << "Can't open file \"" << strF_file_name.str() << "\"" << std::endl;
        exit(-1);
    }
    // tecplot header
    out_file << "<?xml version=\"1.0\"?>\n"
             << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    out_file << "<ImageData WholeExtent="
             << "\"0 " << rhoF.GetNX()-1 << " 0 " << rhoF.GetNY()-1 << " 0 0\" "
             << "Origin=\"0 0 0\" Spacing=\"1 1 1\" Direction=\"1 0 0 0 1 0 0 0 1\">\n";
    out_file << "<Piece Extent=\"0 " << rhoF.GetNX()-1 << " 0 " << rhoF.GetNY()-1 << " 0 0\">\n";
    out_file << "<PointData>\n";

    out_file << "<DataArray type=\"Float64\" Name=\"rho\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int j=0; j<rhoF.GetNY(); ++j) {
        for (int i=0; i<rhoF.GetNX(); ++i) {
            out_file << rhoF(i,j) << "\n";
        }
    }
    out_file << "</DataArray>" << std::endl;

    out_file << "<DataArray type=\"Float64\" Name=\"vel\" NumberOfComponents=\"2\" format=\"ascii\">\n";
    for (int j=0; j<velF.GetNY(); ++j) {
        for (int i=0; i<velF.GetNX(); ++i) {
            out_file << velF(i,j) << "\n";
        }
    }
    out_file << "</DataArray>" << std::endl;

    out_file << "<DataArray type=\"Float64\" Name=\"strF\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    for (int j=0; j<strF.GetNY(); ++j) {
        for (int i=0; i<strF.GetNX(); ++i) {
            out_file << strF(i,j) << "\n";
        }
    }
    out_file << "</DataArray>" << std::endl;

    out_file << "</PointData>\n";
    out_file << "</Piece>\n";
    out_file << "</ImageData>\n";
    out_file << "</VTKFile>" << std::endl;
}
