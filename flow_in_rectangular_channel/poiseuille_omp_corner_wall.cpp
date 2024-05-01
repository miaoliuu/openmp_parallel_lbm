/** 该程序用于计算恒定速度入口流
 *  
 *  注意 NEEM 方法一定要先计算边，然后计算角点
 *  这样做是为了确保计算角点时不会取到边上未更新的分布函数值
 * 
 *  y = 0 恒速度入口，y == ny-1 恒压力出口且流向速度充分发展切向速度为 0 
 * 
 *  该版本的出/入口 x 从 1 到 NX-1 故先处理出入口，再处理 wall
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
#include "src/D2Q9BGK.hpp"
#include <ctime>
#include <cmath>
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
LBM::LatticeField latticeF, latticeF0;

T feq_i(int i, LBM::Density_t rho, LBM::Velocity_t vel, T uSqr);
void init(SimParam const& param);
void evolution(SimParam const& param);
T ErrorVel();
T AverageRho();
void writeData(int t);
void write_stream_func(int t);

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
    int resolution = 20;
    T lx = 1;
    T ly = 5;
    SimParam const ldc_param(Re, la_Umax, la_Cs, resolution, lx, ly);
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
                    write_stream_func(t);
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
    uci = LBM::c[i][0]*u[0] + LBM::c[i][1]*u[1];
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
    rhoF = LBM::DensityField(nx,ny);
    velF = LBM::VelocityField(nx,ny);
    velF0 = LBM::VelocityField(nx,ny);
    latticeF = LBM::LatticeField(nx,ny);
    latticeF0 = LBM::LatticeField(nx,ny);

    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            rhoF(x,y) = rho0;
            velF(x,y)[0] = 0.0;
            velF(x,y)[1] = 0.0;
            velF(x,0)[1] = U0; // y == 0 处为速度入口
            velF(0,0)[1] = 0.0; // 入口角点速度为 0
            velF(nx-1,0)[1] = 0.0; // 入口角点速度为 0
            T uSqr = pow(velF(x,y)[0],2) + pow(velF(x,y)[1],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,y)[i] = feq_i(i, rhoF(x,y), velF(x,y), uSqr);
            }
        }
    }
}

void evolution(SimParam const& param)
{
    int const nx = param.GetNx();
    int const ny = param.GetNy();
    T const omega = param.GetOmega();
    // Bulk
    // ---- collision and streaming (separate) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int y=0; y<ny; ++y) {
        for (int x=0; x<nx; ++x) {
            T uSqr = pow(velF(x,y)[0],2) + pow(velF(x,y)[1],2);
            for (int i=0; i<LBM::q; ++i) {
                latticeF(x,y)[i] *= (1.0-omega);
                latticeF(x,y)[i] += omega*feq_i(i, rhoF(x,y), velF(x,y), uSqr);
            }
        }
    }
    // ---- streaming just bulk (boundary not streaming) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int y=1; y<ny-1; ++y) {
        for (int x=1; x<nx-1; ++x) {
            for (int i=0; i<LBM::q; ++i) {
                int ix = x - LBM::c[i][0];
                int iy = y - LBM::c[i][1];
                latticeF0(x,y)[i] = latticeF(ix,iy)[i];
            }
        }
    }
    // ---- update Micro of bulk ----
    latticeF.Swap(latticeF0);
    // ---- Update Macro of bulk ----
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
            rhoF(x,y) = m0; // Update rhoF
        #ifdef MY_INCOMPRESSIBLE
            velF(x,y)[0] = m1_x;
            velF(x,y)[1] = m1_y;
        #else
            velF(x,y)[0] = m1_x / m0;
            velF(x,y)[1] = m1_y / m0;
        #endif
        }
    }
    // boundary (NEEM)
    // ---- edges (X-direction) ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int x=1; x<nx-1; ++x) {
        T uSqr_1; // uSqr_1 for boundary
        T uSqr_2; // uSqr_2 for bulk
        // ---- bottom ----
        rhoF(x,0) = rhoF(x,1);
        uSqr_1 = pow(velF(x,0)[0],2) + pow(velF(x,0)[1],2);
        uSqr_2 = pow(velF(x,1)[0],2) + pow(velF(x,1)[1],2);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(x,0)[i] = feq_i(i, rhoF(x,0), velF(x,0), uSqr_1);
            latticeF(x,0)[i] += latticeF(x,1)[i] - feq_i(i, rhoF(x,1), velF(x,1), uSqr_2);
        }
        // ---- top ----
        rhoF(x,ny-1) = 1.0;
        velF(x,ny-1)[1] = velF(x,ny-2)[1];
        uSqr_1 = pow(velF(x,ny-1)[0],2) + pow(velF(x,ny-1)[1],2);
        uSqr_2 = pow(velF(x,ny-2)[0],2) + pow(velF(x,ny-2)[1],2);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(x,ny-1)[i] = feq_i(i, rhoF(x,ny-1), velF(x,ny-1), uSqr_1);
            latticeF(x,ny-1)[i] += latticeF(x,ny-2)[i] - feq_i(i, rhoF(x,ny-2), velF(x,ny-2), uSqr_2);
        }
    }
    // ---- edges (Y-direction) with corner ----
#ifdef MY_OMP_OPEN
    #pragma omp parallel for
#endif
    for (int y=0; y<ny; ++y) {
        T uSqr_1; // for boundary
        T uSqr_2; // for bulk
        // ---- left ----
        rhoF(0,y) = rhoF(1,y);
        uSqr_1 = pow(velF(0,y)[0],2) + pow(velF(0,y)[1],2);
        uSqr_2 = pow(velF(1,y)[0],2) + pow(velF(1,y)[1],2);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(0,y)[i] = feq_i(i, rhoF(0,y), velF(0,y), uSqr_1);
            latticeF(0,y)[i] += latticeF(1,y)[i] - feq_i(i, rhoF(1,y), velF(1,y), uSqr_2);
        }
        // ---- right ----
        rhoF(nx-1,y) = rhoF(nx-2,y);
        uSqr_1 = pow(velF(nx-1,y)[0],2) + pow(velF(nx-1,y)[1],2);
        uSqr_2 = pow(velF(nx-2,y)[0],2) + pow(velF(nx-2,y)[1],2);
        for (int i=0; i<LBM::q; ++i) {
            latticeF(nx-1,y)[i] = feq_i(i, rhoF(nx-1,y), velF(nx-1,y), uSqr_1);
            latticeF(nx-1,y)[i] += latticeF(nx-2,y)[i] - feq_i(i, rhoF(nx-2,y), velF(nx-2,y), uSqr_2);
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
    file_name << "poiseuille_" << t << ".vti";
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
    strF_file_name << "poiseuille_strF_" << t << ".vti";
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
