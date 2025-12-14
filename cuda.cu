#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <mpi.h>
#include <cuda_runtime.h>

using namespace std;

const double Lx = M_PI;
const double Ly = M_PI;
const double Lz = M_PI;
const double T = 0.1;
const int N = 128;
const int K = 100;

const double hx = Lx / (N - 1);
const double hy = Ly / (N - 1);
const double hz = Lz / (N - 1);
const double tau = T / K;
const double hx2 = hx * hx;
const double hy2 = hy * hy;
const double hz2 = hz * hz;
const double tau2 = tau * tau;

const double at_val = 2.0 * M_PI;
const double a_squared = 4.0 / (9.0/(Lx*Lx) + 4.0/(Ly*Ly) + 4.0/(Lz*Lz));

const double a2_hx2 = a_squared / hx2;
const double a2_hy2 = a_squared / hy2;
const double a2_hz2 = a_squared / hz2;

int mpi_rank, mpi_size;
int mpi_dims[3];
int mpi_coords[3];
int local_nx, local_ny, local_nz;
int start_x, start_y, start_z;
MPI_Comm cart_comm;


struct TimingData {
    double cuda_init = 0.0;
    double mem_alloc = 0.0;
    double init_kernel = 0.0;
    double error_computation = 0.0;
    double communication = 0.0;
    double comm_copy_host = 0.0;
    double comm_mpi = 0.0;
    double comm_copy_device = 0.0;
    double laplacian_kernel = 0.0;
    double step_kernel = 0.0;
    double boundary_kernel = 0.0;
    double total_program = 0.0;
    
    void reset() {
        *this = TimingData();
    }
    
    void print() const {
        cout << "\nЗАМЕРЫ ВРЕМЕНИ:" << endl;
        cout << "Инициализация CUDA: " << cuda_init << " сек" << endl;
        cout << "Выделение памяти: " << mem_alloc << " сек" << endl;
        cout << "Инициализация сетки: " << init_kernel << " сек" << endl;
        cout << "Вычисление ошибки: " << error_computation << " сек" << endl;
        cout << "Обмен границами: " << communication << " сек" << endl;
        cout << "  Копирование на хост: " << comm_copy_host << " сек" << endl;
        cout << "  MPI обмен: " << comm_mpi << " сек" << endl;
        cout << "  Копирование на устройство: " << comm_copy_device << " сек" << endl;
        cout << "Ядро лапласиана: " << laplacian_kernel << " сек" << endl;
        cout << "Ядро шага по времени: " << step_kernel << " сек" << endl;
        cout << "Ядро граничных условий: " << boundary_kernel << " сек" << endl;
    }
};

TimingData timing;

struct DeviceGrid {
    double* data;
    int nx, ny, nz;
    int total_size;
    
    DeviceGrid(int nx_, int ny_, int nz_) : nx(nx_), ny(ny_), nz(nz_) {
        total_size = nx * ny * nz;
        cudaMalloc(&data, total_size * sizeof(double));
        cudaMemset(data, 0, total_size * sizeof(double));
    }
    
    ~DeviceGrid() {
        if (data) cudaFree(data);
    }
    
    void swapData(DeviceGrid& other) {
        std::swap(data, other.data);
        std::swap(total_size, other.total_size);
    }
};

struct DeviceBuffers {
    double* left_send, *left_recv;
    double* right_send, *right_recv;
    double* bottom_send, *bottom_recv;
    double* top_send, *top_recv;
    double* back_send, *back_recv;
    double* front_send, *front_recv;
    
    int left_right_size;
    int bottom_top_size;
    int back_front_size;
    
    DeviceBuffers(int lr_size, int bt_size, int bf_size) : 
        left_right_size(lr_size), bottom_top_size(bt_size), back_front_size(bf_size) {
        cudaMalloc(&left_send, lr_size * sizeof(double));
        cudaMalloc(&left_recv, lr_size * sizeof(double));
        cudaMalloc(&right_send, lr_size * sizeof(double));
        cudaMalloc(&right_recv, lr_size * sizeof(double));
        
        cudaMalloc(&bottom_send, bt_size * sizeof(double));
        cudaMalloc(&bottom_recv, bt_size * sizeof(double));
        cudaMalloc(&top_send, bt_size * sizeof(double));
        cudaMalloc(&top_recv, bt_size * sizeof(double));
        
        cudaMalloc(&back_send, bf_size * sizeof(double));
        cudaMalloc(&back_recv, bf_size * sizeof(double));
        cudaMalloc(&front_send, bf_size * sizeof(double));
        cudaMalloc(&front_recv, bf_size * sizeof(double));
    }
    
    ~DeviceBuffers() {
        if (left_send) cudaFree(left_send);
        if (left_recv) cudaFree(left_recv);
        if (right_send) cudaFree(right_send);
        if (right_recv) cudaFree(right_recv);
        if (bottom_send) cudaFree(bottom_send);
        if (bottom_recv) cudaFree(bottom_recv);
        if (top_send) cudaFree(top_send);
        if (top_recv) cudaFree(top_recv);
        if (back_send) cudaFree(back_send);
        if (back_recv) cudaFree(back_recv);
        if (front_send) cudaFree(front_send);
        if (front_recv) cudaFree(front_recv);
    }
};

__global__ void initGridKernel(double* u, int nx, int ny, int nz,
                              int start_x, int start_y, int start_z,
                              double hx, double hy, double hz,
                              double Lx, double Ly, double Lz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz) {
        double x = (start_x + i) * hx;
        double y = (start_y + j) * hy;
        double z = (start_z + k) * hz;
        
        double value = sin(3.0 * M_PI * x / Lx) * 
                      sin(2.0 * M_PI * y / Ly) * 
                      sin(2.0 * M_PI * z / Lz);
        
        int idx = i * ny * nz + j * nz + k;
        u[idx] = value;
    }
}

__global__ void copyBoundaryToBufferKernel(double* u, 
                                          double* left_buf, double* right_buf,
                                          double* bottom_buf, double* top_buf,
                                          double* back_buf, double* front_buf,
                                          int nx, int ny, int nz) {
  
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (j < ny && k < nz) {
        int src_idx = 0 * ny * nz + j * nz + k;
        int dst_idx = j * nz + k;
        left_buf[dst_idx] = u[src_idx];
    }
    
    if (j < ny && k < nz) {
        int src_idx = (nx-1) * ny * nz + j * nz + k;
        int dst_idx = j * nz + k;
        right_buf[dst_idx] = u[src_idx];
    }

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && k < nz) {
        int src_idx = i * ny * nz + 0 * nz + k;
        int dst_idx = i * nz + k;
        bottom_buf[dst_idx] = u[src_idx];
    }
    
    if (i < nx && k < nz) {
        int src_idx = i * ny * nz + (ny-1) * nz + k;
        int dst_idx = i * nz + k;
        top_buf[dst_idx] = u[src_idx];
    }
    
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny) {
        int src_idx = i * ny * nz + j * nz + 0;
        int dst_idx = i * ny + j;
        back_buf[dst_idx] = u[src_idx];
    }
    
    if (i < nx && j < ny) {
        int src_idx = i * ny * nz + j * nz + (nz-1);
        int dst_idx = i * ny + j;
        front_buf[dst_idx] = u[src_idx];
    }
}

__global__ void applyBoundaryFromBufferKernel(double* u,
                                             double* left_buf, double* right_buf,
                                             double* bottom_buf, double* top_buf,
                                             double* back_buf, double* front_buf,
                                             int nx, int ny, int nz,
                                             int start_x, int start_y, int start_z,
                                             int global_N, int mpi_dims_x, int mpi_dims_y, int mpi_dims_z) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (j < ny && k < nz) {
        int dst_idx = 0 * ny * nz + j * nz + k;
        int src_idx = j * nz + k;
        u[dst_idx] = left_buf[src_idx];
    }
    
    if (j < ny && k < nz) {
        int dst_idx = (nx-1) * ny * nz + j * nz + k;
        int src_idx = j * nz + k;
        u[dst_idx] = right_buf[src_idx];
    }
    
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && k < nz) {
        int dst_idx = i * ny * nz + 0 * nz + k;
        int src_idx = i * nz + k;
        u[dst_idx] = bottom_buf[src_idx];
    }
    
    if (i < nx && k < nz) {
        int dst_idx = i * ny * nz + (ny-1) * nz + k;
        int src_idx = i * nz + k;
        u[dst_idx] = top_buf[src_idx];
    }
    
    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny) {
        int dst_idx = i * ny * nz + j * nz + 0;
        int src_idx = i * ny + j;
        u[dst_idx] = back_buf[src_idx];
    }
    
    if (i < nx && j < ny) {
        int dst_idx = i * ny * nz + j * nz + (nz-1);
        int src_idx = i * ny + j;
        u[dst_idx] = front_buf[src_idx];
    }
}

__global__ void computeLaplacianKernel(double* u, double* laplacian,
                                      int nx, int ny, int nz,
                                      double a2_hx2, double a2_hy2, double a2_hz2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
        int idx = i * ny * nz + j * nz + k;
        
        double u_ijk = u[idx];
        
        double dx = (u[(i-1)*ny*nz + j*nz + k] - 2*u_ijk + u[(i+1)*ny*nz + j*nz + k]) * a2_hx2;
        double dy = (u[i*ny*nz + (j-1)*nz + k] - 2*u_ijk + u[i*ny*nz + (j+1)*nz + k]) * a2_hy2;
        double dz = (u[i*ny*nz + j*nz + (k-1)] - 2*u_ijk + u[i*ny*nz + j*nz + (k+1)]) * a2_hz2;
        
        laplacian[idx] = dx + dy + dz;
    }
}

__global__ void firstStepKernel(double* u0, double* u1, double* laplacian,
                               int nx, int ny, int nz,
                               int start_x, int start_y, int start_z,
                               int global_N,
                               double tau2_half) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz) {
        int idx = i * ny * nz + j * nz + k;
       
        if ((i == 0 && start_x == 0) || (i == nx-1 && (start_x + nx) == global_N)) {
            u1[idx] = 0.0;
        } 
        else if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
            u1[idx] = u0[idx] + tau2_half * laplacian[idx];
        } else {
            u1[idx] = u0[idx];
        }
    }
}

__global__ void timeStepKernel(double* u0, double* u1, double* u2, double* laplacian,
                              int nx, int ny, int nz,
                              int start_x, int start_y, int start_z,
                              int global_N,
                              double tau2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz) {
        int idx = i * ny * nz + j * nz + k;
        
        if ((i == 0 && start_x == 0) || (i == nx-1 && (start_x + nx) == global_N)) {
            u2[idx] = 0.0;
        } else if (i > 0 && i < nx-1 && j > 0 && j < ny-1 && k > 0 && k < nz-1) {
            u2[idx] = 2 * u1[idx] - u0[idx] + tau2 * laplacian[idx];
        } else {
            u2[idx] = u1[idx];
        }
    }
}

__global__ void errorKernel(const double* u, double* max_error,
                           int nx, int ny, int nz,
                           int start_x, int start_y, int start_z,
                           double hx, double hy, double hz,
                           double t, double Lx, double Ly, double Lz,
                           double at_val) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (i < nx && j < ny && k < nz) {
        double x = (start_x + i) * hx;
        double y = (start_y + j) * hy;
        double z = (start_z + k) * hz;
        
        double u_exact = sin(3.0 * M_PI * x / Lx) * 
                        sin(2.0 * M_PI * y / Ly) * 
                        sin(2.0 * M_PI * z / Lz) * 
                        cos(at_val * t + 4.0 * M_PI);
        
        int idx = i * ny * nz + j * nz + k;
        double error = fabs(u[idx] - u_exact);
        
        unsigned long long int* address_as_ull = (unsigned long long int*)max_error;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            if (__longlong_as_double(assumed) >= error)
                break;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(error));
        } while (assumed != old);
    }
}

void init_mpi(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    mpi_dims[0] = mpi_dims[1] = mpi_dims[2] = 0;
    MPI_Dims_create(mpi_size, 3, mpi_dims);
    
    int periods[3] = {0, 1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 3, mpi_dims, periods, 1, &cart_comm);
    MPI_Comm_rank(cart_comm, &mpi_rank);
    MPI_Cart_coords(cart_comm, mpi_rank, 3, mpi_coords);
    
    local_nx = N / mpi_dims[0];
    local_ny = N / mpi_dims[1];
    local_nz = N / mpi_dims[2];
    
    start_x = mpi_coords[0] * local_nx;
    start_y = mpi_coords[1] * local_ny;
    start_z = mpi_coords[2] * local_nz;
    
    if (mpi_coords[0] == mpi_dims[0] - 1) local_nx = N - start_x;
    if (mpi_coords[1] == mpi_dims[1] - 1) local_ny = N - start_y;
    if (mpi_coords[2] == mpi_dims[2] - 1) local_nz = N - start_z;
}

void get_neighbors(int& left, int& right, int& bottom, int& top, int& back, int& front) {
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &bottom, &top);
    MPI_Cart_shift(cart_comm, 2, 1, &back, &front);
}

void exchange_boundaries(DeviceGrid& u, DeviceBuffers& buffers) {
    double total_time = MPI_Wtime();
    
    int left, right, bottom, top, back, front;
    get_neighbors(left, right, bottom, top, back, front);
    
    dim3 blockSize(4, 8, 8);
    dim3 gridSizeLR(1, (local_ny + blockSize.y - 1) / blockSize.y, 
                         (local_nz + blockSize.z - 1) / blockSize.z);
    
    double copy_time = MPI_Wtime();
    copyBoundaryToBufferKernel<<<gridSizeLR, blockSize>>>(u.data,
        buffers.left_send, buffers.right_send,
        buffers.bottom_send, buffers.top_send,
        buffers.back_send, buffers.front_send,
        local_nx, local_ny, local_nz);
    cudaDeviceSynchronize();
    timing.boundary_kernel += MPI_Wtime() - copy_time;
    
    vector<double> left_send_host(buffers.left_right_size);
    vector<double> right_send_host(buffers.left_right_size);
    vector<double> bottom_send_host(buffers.bottom_top_size);
    vector<double> top_send_host(buffers.bottom_top_size);
    vector<double> back_send_host(buffers.back_front_size);
    vector<double> front_send_host(buffers.back_front_size);
    
    vector<double> left_recv_host(buffers.left_right_size);
    vector<double> right_recv_host(buffers.left_right_size);
    vector<double> bottom_recv_host(buffers.bottom_top_size);
    vector<double> top_recv_host(buffers.bottom_top_size);
    vector<double> back_recv_host(buffers.back_front_size);
    vector<double> front_recv_host(buffers.back_front_size);
    
    double copy_to_host_time = MPI_Wtime();
    cudaMemcpy(left_send_host.data(), buffers.left_send, buffers.left_right_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(right_send_host.data(), buffers.right_send, buffers.left_right_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(bottom_send_host.data(), buffers.bottom_send, buffers.bottom_top_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(top_send_host.data(), buffers.top_send, buffers.bottom_top_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(back_send_host.data(), buffers.back_send, buffers.back_front_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(front_send_host.data(), buffers.front_send, buffers.back_front_size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    timing.comm_copy_host += MPI_Wtime() - copy_to_host_time;
    
    double mpi_time = MPI_Wtime();
    MPI_Request requests[12];
    int req_count = 0;
    
    if (left != MPI_PROC_NULL) {
        MPI_Isend(left_send_host.data(), buffers.left_right_size, MPI_DOUBLE, left, 0, cart_comm, &requests[req_count++]);
        MPI_Irecv(left_recv_host.data(), buffers.left_right_size, MPI_DOUBLE, left, 1, cart_comm, &requests[req_count++]);
    }
    
    if (right != MPI_PROC_NULL) {
        MPI_Isend(right_send_host.data(), buffers.left_right_size, MPI_DOUBLE, right, 1, cart_comm, &requests[req_count++]);
        MPI_Irecv(right_recv_host.data(), buffers.left_right_size, MPI_DOUBLE, right, 0, cart_comm, &requests[req_count++]);
    }
    
    if (bottom != MPI_PROC_NULL) {
        MPI_Isend(bottom_send_host.data(), buffers.bottom_top_size, MPI_DOUBLE, bottom, 2, cart_comm, &requests[req_count++]);
        MPI_Irecv(bottom_recv_host.data(), buffers.bottom_top_size, MPI_DOUBLE, bottom, 3, cart_comm, &requests[req_count++]);
    }
    
    if (top != MPI_PROC_NULL) {
        MPI_Isend(top_send_host.data(), buffers.bottom_top_size, MPI_DOUBLE, top, 3, cart_comm, &requests[req_count++]);
        MPI_Irecv(top_recv_host.data(), buffers.bottom_top_size, MPI_DOUBLE, top, 2, cart_comm, &requests[req_count++]);
    }
    
    if (back != MPI_PROC_NULL) {
        MPI_Isend(back_send_host.data(), buffers.back_front_size, MPI_DOUBLE, back, 4, cart_comm, &requests[req_count++]);
        MPI_Irecv(back_recv_host.data(), buffers.back_front_size, MPI_DOUBLE, back, 5, cart_comm, &requests[req_count++]);
    }
    
    if (front != MPI_PROC_NULL) {
        MPI_Isend(front_send_host.data(), buffers.back_front_size, MPI_DOUBLE, front, 5, cart_comm, &requests[req_count++]);
        MPI_Irecv(front_recv_host.data(), buffers.back_front_size, MPI_DOUBLE, front, 4, cart_comm, &requests[req_count++]);
    }
    
    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
    timing.comm_mpi += MPI_Wtime() - mpi_time;
    
    double copy_to_device_time = MPI_Wtime();
    cudaMemcpy(buffers.left_recv, left_recv_host.data(), buffers.left_right_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers.right_recv, right_recv_host.data(), buffers.left_right_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers.bottom_recv, bottom_recv_host.data(), buffers.bottom_top_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers.top_recv, top_recv_host.data(), buffers.bottom_top_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers.back_recv, back_recv_host.data(), buffers.back_front_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(buffers.front_recv, front_recv_host.data(), buffers.back_front_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    timing.comm_copy_device += MPI_Wtime() - copy_to_device_time;
    
    double apply_time = MPI_Wtime();
    applyBoundaryFromBufferKernel<<<gridSizeLR, blockSize>>>(u.data,
        buffers.left_recv, buffers.right_recv,
        buffers.bottom_recv, buffers.top_recv,
        buffers.back_recv, buffers.front_recv,
        local_nx, local_ny, local_nz,
        start_x, start_y, start_z, N,
        mpi_dims[0], mpi_dims[1], mpi_dims[2]);
    cudaDeviceSynchronize();
    timing.boundary_kernel += MPI_Wtime() - apply_time;
    
    timing.communication += MPI_Wtime() - total_time;
}

double compute_error(DeviceGrid& u, double t, int start_x, int start_y, int start_z) {
    double error_time = MPI_Wtime();
    
    double* d_max_error;
    cudaMalloc(&d_max_error, sizeof(double));
    double zero = 0.0;
    cudaMemcpy(d_max_error, &zero, sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 blockSize(4, 4, 4);
    dim3 gridSize((local_nx + blockSize.x - 1) / blockSize.x,
                  (local_ny + blockSize.y - 1) / blockSize.y,
                  (local_nz + blockSize.z - 1) / blockSize.z);
    
    errorKernel<<<gridSize, blockSize>>>(u.data, d_max_error,
                                        local_nx, local_ny, local_nz,
                                        start_x, start_y, start_z,
                                        hx, hy, hz, t,
                                        Lx, Ly, Lz, at_val);
    cudaDeviceSynchronize();
    
    double local_max;
    cudaMemcpy(&local_max, d_max_error, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_max_error);
    
    double global_max;
    MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    
    timing.error_computation += MPI_Wtime() - error_time;
    
    return global_max;
}

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);
    double program_start = MPI_Wtime();
    
    if (mpi_rank == 0) {
        cout << "MPI + CUDA Wave Equation Solver" << endl;
        cout << "Grid: " << N << "^3, Time steps: " << K << endl;
        cout << "MPI processes: " << mpi_size << " (" << mpi_dims[0] 
             << "x" << mpi_dims[1] << "x" << mpi_dims[2] << ")" << endl;
        cout << "Local grid: " << local_nx << "x" << local_ny << "x" << local_nz << endl;
    }
    
    timing.cuda_init = MPI_Wtime();
    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(mpi_rank % device_count);
    timing.cuda_init = MPI_Wtime() - timing.cuda_init;
    
    timing.mem_alloc = MPI_Wtime();
    DeviceGrid u0(local_nx, local_ny, local_nz);
    DeviceGrid u1(local_nx, local_ny, local_nz);
    DeviceGrid u2(local_nx, local_ny, local_nz);
    DeviceGrid laplacian(local_nx, local_ny, local_nz);
    DeviceBuffers buffers(local_ny * local_nz, local_nx * local_nz, local_nx * local_ny);
    timing.mem_alloc = MPI_Wtime() - timing.mem_alloc;
    
    dim3 blockSize(8, 8, 4);
    dim3 gridSize((local_nx + blockSize.x - 1) / blockSize.x,
                  (local_ny + blockSize.y - 1) / blockSize.y,
                  (local_nz + blockSize.z - 1) / blockSize.z);
    
    timing.init_kernel = MPI_Wtime();
    initGridKernel<<<gridSize, blockSize>>>(u0.data, local_nx, local_ny, local_nz,
                                           start_x, start_y, start_z,
                                           hx, hy, hz, Lx, Ly, Lz);
    cudaDeviceSynchronize();
    timing.init_kernel = MPI_Wtime() - timing.init_kernel;
    
    double error0 = compute_error(u0, 0.0, start_x, start_y, start_z);
    
    if (mpi_rank == 0) {
        cout << "Initial error: " << error0 << endl;
    }
    
    exchange_boundaries(u0, buffers);
    
    timing.laplacian_kernel = MPI_Wtime();
    computeLaplacianKernel<<<gridSize, blockSize>>>(u0.data, laplacian.data,
                                                   local_nx, local_ny, local_nz,
                                                   a2_hx2, a2_hy2, a2_hz2);
    cudaDeviceSynchronize();
    timing.laplacian_kernel = MPI_Wtime() - timing.laplacian_kernel;
    
    timing.step_kernel = MPI_Wtime();
    const double tau2_half = tau2 * 0.5;
    firstStepKernel<<<gridSize, blockSize>>>(u0.data, u1.data, laplacian.data,
                                            local_nx, local_ny, local_nz,
                                            start_x, start_y, start_z, N,
                                            tau2_half);
    cudaDeviceSynchronize();
    timing.step_kernel = MPI_Wtime() - timing.step_kernel;
    
    exchange_boundaries(u1, buffers);
    
    double error1 = compute_error(u1, tau, start_x, start_y, start_z);
    
    if (mpi_rank == 0) {
        cout << "Step 1, t = " << tau << ", error = " << error1 << endl;
    }
    
    vector<double> errors;
    errors.push_back(error0);
    errors.push_back(error1);
    
    for (int n = 1; n < K; n++) {
        double lap_start = MPI_Wtime();
        computeLaplacianKernel<<<gridSize, blockSize>>>(u1.data, laplacian.data,
                                                       local_nx, local_ny, local_nz,
                                                       a2_hx2, a2_hy2, a2_hz2);
        cudaDeviceSynchronize();
        timing.laplacian_kernel += MPI_Wtime() - lap_start;
        
        double step_start = MPI_Wtime();
        timeStepKernel<<<gridSize, blockSize>>>(u0.data, u1.data, u2.data, laplacian.data,
                                               local_nx, local_ny, local_nz,
                                               start_x, start_y, start_z, N,
                                               tau2);
        cudaDeviceSynchronize();
        timing.step_kernel += MPI_Wtime() - step_start;
        
        exchange_boundaries(u2, buffers);
        
        u0.swapData(u1);
        u1.swapData(u2);
        
        if (n % 10 == 0) {
            double current_time = (n + 1) * tau;
            double current_error = compute_error(u1, current_time, start_x, start_y, start_z);
            errors.push_back(current_error);
            
            if (mpi_rank == 0) {
                cout << "Step " << (n + 1) << ", t = " << current_time 
                     << ", error = " << current_error << endl;
            }
        }
    }
    
    double final_error = compute_error(u1, T, start_x, start_y, start_z);
    
    timing.total_program = MPI_Wtime() - program_start;
    
    double timing_data[12];
    timing_data[0] = timing.cuda_init;
    timing_data[1] = timing.mem_alloc;
    timing_data[2] = timing.init_kernel;
    timing_data[3] = timing.error_computation;
    timing_data[4] = timing.communication;
    timing_data[5] = timing.comm_copy_host;
    timing_data[6] = timing.comm_mpi;
    timing_data[7] = timing.comm_copy_device;
    timing_data[8] = timing.laplacian_kernel;
    timing_data[9] = timing.step_kernel;
    timing_data[10] = timing.boundary_kernel;
    timing_data[11] = timing.total_program;
    
    double global_timing_data[12] = {0};
    
    MPI_Reduce(timing_data, global_timing_data, 11, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    MPI_Reduce(&timing_data[11], &global_timing_data[11], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    double final_error_global;
    MPI_Reduce(&final_error, &final_error_global, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (mpi_rank == 0) {
        TimingData global_timing;
        
        global_timing.cuda_init = global_timing_data[0] / mpi_size;
        global_timing.mem_alloc = global_timing_data[1] / mpi_size;
        global_timing.init_kernel = global_timing_data[2] / mpi_size;
        global_timing.error_computation = global_timing_data[3] / mpi_size;
        global_timing.communication = global_timing_data[4] / mpi_size;
        global_timing.comm_copy_host = global_timing_data[5] / mpi_size;
        global_timing.comm_mpi = global_timing_data[6] / mpi_size;
        global_timing.comm_copy_device = global_timing_data[7] / mpi_size;
        global_timing.laplacian_kernel = global_timing_data[8] / mpi_size;
        global_timing.step_kernel = global_timing_data[9] / mpi_size;
        global_timing.boundary_kernel = global_timing_data[10] / mpi_size;
        global_timing.total_program = global_timing_data[11];
        
        global_timing.print();
        
        cout << "Полученная погрешность: " << final_error_global << endl;
        cout << "Общее время работы программы: " << global_timing.total_program << " seconds" << endl;
        
    }
    
    MPI_Finalize();
    return 0;
}