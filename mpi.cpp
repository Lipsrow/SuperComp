#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <mpi.h>

using namespace std;

const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;
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

double u_analytical(double x, double y, double z, double t) {
    double factor1 = 3.0 * M_PI / Lx;
    double factor2 = 2.0 * M_PI / Ly;
    double factor3 = 2.0 * M_PI / Lz;
    double at_val = 2.0 * M_PI;
    
    return sin(factor1 * x) * cos(factor2 * y) * cos(factor3 * z) * cos(at_val * t + 4.0 * M_PI);
}

double phi(double x, double y, double z) {
    return u_analytical(x, y, z, 0.0);
}

class Grid3D {
private:
    vector<double> data;
    int nx, ny, nz;
    
public:
    Grid3D(int nx, int ny, int nz) : nx(nx), ny(ny), nz(nz), data(nx * ny * nz, 0.0) {}
    
    double& operator()(int i, int j, int k) {
        return data[i * ny * nz + j * nz + k];
    }
    
    double operator()(int i, int j, int k) const {
        return data[i * ny * nz + j * nz + k];
    }
    
    void fill(double value) {
        std::fill(data.begin(), data.end(), value);
    }
    
    int size() const { return nx * ny * nz; }
    int get_nx() const { return nx; }
    int get_ny() const { return ny; }
    int get_nz() const { return nz; }
    
    const vector<double>& get_data() const { return data; }
};

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

void finalize_mpi() {
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
}

void get_neighbors(int& left, int& right, int& bottom, int& top, int& back, int& front) {
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &bottom, &top);
    MPI_Cart_shift(cart_comm, 2, 1, &back, &front);
}

void initialize_grid(Grid3D& grid) {
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                double x = (start_x + i) * hx;
                double y = (start_y + j) * hy;
                double z = (start_z + k) * hz;
                grid(i, j, k) = phi(x, y, z);
            }
        }
    }
    
    if (start_x == 0) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                grid(0, j, k) = 0.0;
            }
        }
    }
    if (start_x + local_nx == N) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                grid(local_nx-1, j, k) = 0.0;
            }
        }
    }
}

void exchange_with_neighbors(Grid3D& grid, 
                     vector<double>& neighbor_left, vector<double>& neighbor_right,
                     vector<double>& neighbor_bottom, vector<double>& neighbor_top,
                     vector<double>& neighbor_back, vector<double>& neighbor_front) {
    
    int left, right, bottom, top, back, front;
    get_neighbors(left, right, bottom, top, back, front);
    
    MPI_Request requests[12];
    int request_count = 0;
    vector<vector<double>> send_buffers;
    
    if (left != MPI_PROC_NULL) {
        vector<double> send_left(local_ny * local_nz);
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                send_left[j * local_nz + k] = grid(0, j, k);
            }
        }
        send_buffers.push_back(send_left);
        MPI_Isend(send_buffers.back().data(), send_left.size(), MPI_DOUBLE, left, 0, cart_comm, &requests[request_count++]);
        MPI_Irecv(neighbor_left.data(), neighbor_left.size(), MPI_DOUBLE, left, 1, cart_comm, &requests[request_count++]);
    } else {
        fill(neighbor_left.begin(), neighbor_left.end(), 0.0);
    }
    
    if (right != MPI_PROC_NULL) {
        vector<double> send_right(local_ny * local_nz);
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                send_right[j * local_nz + k] = grid(local_nx-1, j, k);
            }
        }
        send_buffers.push_back(send_right);
        MPI_Isend(send_buffers.back().data(), send_right.size(), MPI_DOUBLE, right, 1, cart_comm, &requests[request_count++]);
        MPI_Irecv(neighbor_right.data(), neighbor_right.size(), MPI_DOUBLE, right, 0, cart_comm, &requests[request_count++]);
    } else {
        fill(neighbor_right.begin(), neighbor_right.end(), 0.0);
    }
    
    if (bottom != MPI_PROC_NULL) {
        vector<double> send_bottom(local_nx * local_nz);
        for (int i = 0; i < local_nx; i++) {
            for (int k = 0; k < local_nz; k++) {
                send_bottom[i * local_nz + k] = grid(i, 0, k);
            }
        }
        send_buffers.push_back(send_bottom);
        MPI_Isend(send_buffers.back().data(), send_bottom.size(), MPI_DOUBLE, bottom, 2, cart_comm, &requests[request_count++]);
        MPI_Irecv(neighbor_bottom.data(), neighbor_bottom.size(), MPI_DOUBLE, bottom, 3, cart_comm, &requests[request_count++]);
    } else {
        fill(neighbor_bottom.begin(), neighbor_bottom.end(), 0.0);
    }
    
    if (top != MPI_PROC_NULL) {
        vector<double> send_top(local_nx * local_nz);
        for (int i = 0; i < local_nx; i++) {
            for (int k = 0; k < local_nz; k++) {
                send_top[i * local_nz + k] = grid(i, local_ny-1, k);
            }
        }
        send_buffers.push_back(send_top);
        MPI_Isend(send_buffers.back().data(), send_top.size(), MPI_DOUBLE, top, 3, cart_comm, &requests[request_count++]);
        MPI_Irecv(neighbor_top.data(), neighbor_top.size(), MPI_DOUBLE, top, 2, cart_comm, &requests[request_count++]);
    } else {
        fill(neighbor_top.begin(), neighbor_top.end(), 0.0);
    }
    
    if (back != MPI_PROC_NULL) {
        vector<double> send_back(local_nx * local_ny);
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < local_ny; j++) {
                send_back[i * local_ny + j] = grid(i, j, 0);
            }
        }
        send_buffers.push_back(send_back);
        MPI_Isend(send_buffers.back().data(), send_back.size(), MPI_DOUBLE, back, 4, cart_comm, &requests[request_count++]);
        MPI_Irecv(neighbor_back.data(), neighbor_back.size(), MPI_DOUBLE, back, 5, cart_comm, &requests[request_count++]);
    } else {
        fill(neighbor_back.begin(), neighbor_back.end(), 0.0);
    }
    
    if (front != MPI_PROC_NULL) {
        vector<double> send_front(local_nx * local_ny);
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < local_ny; j++) {
                send_front[i * local_ny + j] = grid(i, j, local_nz-1);
            }
        }
        send_buffers.push_back(send_front);
        MPI_Isend(send_buffers.back().data(), send_front.size(), MPI_DOUBLE, front, 5, cart_comm, &requests[request_count++]);
        MPI_Irecv(neighbor_front.data(), neighbor_front.size(), MPI_DOUBLE, front, 4, cart_comm, &requests[request_count++]);
    } else {
        fill(neighbor_front.begin(), neighbor_front.end(), 0.0);
    }
    
    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
}

double laplacian(const Grid3D& u, int i, int j, int k, 
                 const vector<double>& neighbor_left, const vector<double>& neighbor_right,
                 const vector<double>& neighbor_bottom, const vector<double>& neighbor_top,
                 const vector<double>& neighbor_back, const vector<double>& neighbor_front) {
    
    double u_im1, u_ip1, u_jm1, u_jp1, u_km1, u_kp1;
    
    if (i == 0) {
        if (start_x == 0) {
            u_im1 = 0.0;
        } else {
            u_im1 = neighbor_left[j * local_nz + k];
        }
    } else {
        u_im1 = u(i-1, j, k);
    }
    
    if (i == local_nx - 1) {
        if (start_x + local_nx == N) {
            u_ip1 = 0.0;
        } else {
            u_ip1 = neighbor_right[j * local_nz + k];
        }
    } else {
        u_ip1 = u(i+1, j, k);
    }
    
    if (j == 0) {
        u_jm1 = neighbor_bottom[i * local_nz + k];
    } else {
        u_jm1 = u(i, j-1, k);
    }
    
    if (j == local_ny - 1) {
        u_jp1 = neighbor_top[i * local_nz + k];
    } else {
        u_jp1 = u(i, j+1, k);
    }
    
    if (k == 0) {
        u_km1 = neighbor_back[i * local_ny + j];
    } else {
        u_km1 = u(i, j, k-1);
    }
    
    if (k == local_nz - 1) {
        u_kp1 = neighbor_front[i * local_ny + j];
    } else {
        u_kp1 = u(i, j, k+1);
    }
    
    double dx = (u_im1 - 2 * u(i, j, k) + u_ip1) * a2_hx2;
    double dy = (u_jm1 - 2 * u(i, j, k) + u_jp1) * a2_hy2;
    double dz = (u_km1 - 2 * u(i, j, k) + u_kp1) * a2_hz2;
    
    return dx + dy + dz;
}

void compute_first_time_step(const Grid3D& u0, Grid3D& u1,
                            const vector<double>& neighbor_left, const vector<double>& neighbor_right,
                            const vector<double>& neighbor_bottom, const vector<double>& neighbor_top,
                            const vector<double>& neighbor_back, const vector<double>& neighbor_front) {
    
    const double tau2_half = tau2 * 0.5;
    
    for (int i = 0; i < local_nx; i++) {
        if ((start_x == 0 && i == 0) || (start_x + local_nx == N && i == local_nx - 1)) {
            continue;
        }
        
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                u1(i, j, k) = u0(i, j, k) + tau2_half * laplacian(u0, i, j, k, 
                    neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);
            }
        }
    }
    
    if (start_x == 0) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                u1(0, j, k) = 0.0;
            }
        }
    }
    if (start_x + local_nx == N) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                u1(local_nx-1, j, k) = 0.0;
            }
        }
    }
}

void compute_time_step(const Grid3D& u0, const Grid3D& u1, Grid3D& u2,
                      const vector<double>& neighbor_left, const vector<double>& neighbor_right,
                      const vector<double>& neighbor_bottom, const vector<double>& neighbor_top,
                      const vector<double>& neighbor_back, const vector<double>& neighbor_front) {
    
    for (int i = 0; i < local_nx; i++) {
        if ((start_x == 0 && i == 0) || (start_x + local_nx == N && i == local_nx - 1)) {
            continue;
        }
        
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                u2(i, j, k) = 2 * u1(i, j, k) - u0(i, j, k) + 
                              tau2 * laplacian(u1, i, j, k, 
                              neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);
            }
        }
    }
    
    if (start_x == 0) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                u2(0, j, k) = 0.0;
            }
        }
    }
    if (start_x + local_nx == N) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                u2(local_nx-1, j, k) = 0.0;
            }
        }
    }
}

double compute_max_error(const Grid3D& numerical, double current_time, double& max_abs_error, double& max_rel_error) {
    double local_max_abs_error = 0.0;
    double local_max_rel_error = 0.0;
    
    for (int i = 0; i < local_nx; i++) {
        for (int j = 0; j < local_ny; j++) {
            for (int k = 0; k < local_nz; k++) {
                double x = (start_x + i) * hx;
                double y = (start_y + j) * hy;
                double z = (start_z + k) * hz;
                double exact = u_analytical(x, y, z, current_time);
                double num = numerical(i, j, k);
                double abs_error = fabs(num - exact);
                double rel_error = 0.0;
                
                if (fabs(exact) > 1e-12) {
                    rel_error = abs_error / fabs(exact);
                }
                
                if (abs_error > local_max_abs_error) {
                    local_max_abs_error = abs_error;
                }
                if (rel_error > local_max_rel_error) {
                    local_max_rel_error = rel_error;
                }
            }
        }
    }
    
    double global_max_abs_error, global_max_rel_error;
    MPI_Allreduce(&local_max_abs_error, &global_max_abs_error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    MPI_Allreduce(&local_max_rel_error, &global_max_rel_error, 1, MPI_DOUBLE, MPI_MAX, cart_comm);
    
    max_abs_error = global_max_abs_error;
    max_rel_error = global_max_rel_error;
    
    return global_max_abs_error;
}

enum DataType {
    NUMERICAL,
    ANALYTICAL,
    ERROR,
    TIME_ERRORS
};

void write_data_to_file(const Grid3D* numerical, DataType data_type, 
                       const string& filename, double current_time = 0.0,
                       const vector<int>* time_steps = nullptr,
                       const vector<double>* time_values = nullptr,
                       const vector<double>* errors = nullptr) {

    if (data_type == TIME_ERRORS) {
        if (mpi_rank == 0 && time_steps && time_values && errors) {
            ofstream file(filename);
            if (!file.is_open()) {
                cerr << "Не удалось открыть файл для записи погрешностей по шагам!" << endl;
                return;
            }
            
            file << "Time_Step Time Max_Absolute_Error" << endl;
            for (size_t i = 0; i < time_steps->size(); i++) {
                file << (*time_steps)[i] << " " << (*time_values)[i] << " " << (*errors)[i] << endl;
            }
            file.close();
        }
        return;
    }
    
    if (mpi_rank == 0) {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Не удалось открыть файл " << filename << " для записи!" << endl;
            return;
        }
        
        switch (data_type) {
            case NUMERICAL:
                file << "i j k numerical" << endl;
                break;
            case ANALYTICAL:
                file << "i j k analytical" << endl;
                break;
            case ERROR:
                file << "i j k abs_error" << endl;
                break;
            default:
                file << "i j k value" << endl;
                break;
        }
        
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < local_ny; j++) {
                for (int k = 0; k < local_nz; k++) {
                    double x = (start_x + i) * hx;
                    double y = (start_y + j) * hy;
                    double z = (start_z + k) * hz;
                    double value = 0.0;
                    
                    switch (data_type) {
                        case NUMERICAL:
                            value = (*numerical)(i, j, k);
                            break;
                        case ANALYTICAL:
                            value = u_analytical(x, y, z, current_time);
                            break;
                        case ERROR:
                            {
                                double exact = u_analytical(x, y, z, current_time);
                                value = fabs((*numerical)(i, j, k) - exact);
                            }
                            break;
                        default:
                            value = 0.0;
                            break;
                    }
                    
                    file << (start_x + i) << " " << (start_y + j) << " " << (start_z + k) << " "
                         << value << endl;
                }
            }
        }

        for (int proc = 1; proc < mpi_size; proc++) {
            int coords[3];
            MPI_Recv(coords, 3, MPI_INT, proc, 0, cart_comm, MPI_STATUS_IGNORE);
            
            int proc_nx, proc_ny, proc_nz;
            MPI_Recv(&proc_nx, 1, MPI_INT, proc, 1, cart_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&proc_ny, 1, MPI_INT, proc, 2, cart_comm, MPI_STATUS_IGNORE);
            MPI_Recv(&proc_nz, 1, MPI_INT, proc, 3, cart_comm, MPI_STATUS_IGNORE);
            
            vector<double> proc_data(proc_nx * proc_ny * proc_nz);
            MPI_Recv(proc_data.data(), proc_data.size(), MPI_DOUBLE, proc, 4, cart_comm, MPI_STATUS_IGNORE);
            
            // Запись данных от процесса
            for (int i = 0; i < proc_nx; i++) {
                for (int j = 0; j < proc_ny; j++) {
                    for (int k = 0; k < proc_nz; k++) {
                        int idx = i * proc_ny * proc_nz + j * proc_nz + k;
                        file << (coords[0] * (N / mpi_dims[0]) + i) << " " 
                             << (coords[1] * (N / mpi_dims[1]) + j) << " " 
                             << (coords[2] * (N / mpi_dims[2]) + k) << " "
                             << proc_data[idx] << endl;
                    }
                }
            }
        }
        
        file.close();
    } else {
        MPI_Send(mpi_coords, 3, MPI_INT, 0, 0, cart_comm);
        MPI_Send(&local_nx, 1, MPI_INT, 0, 1, cart_comm);
        MPI_Send(&local_ny, 1, MPI_INT, 0, 2, cart_comm);
        MPI_Send(&local_nz, 1, MPI_INT, 0, 3, cart_comm);
        
        vector<double> data(local_nx * local_ny * local_nz);
        for (int i = 0; i < local_nx; i++) {
            for (int j = 0; j < local_ny; j++) {
                for (int k = 0; k < local_nz; k++) {
                    double x = (start_x + i) * hx;
                    double y = (start_y + j) * hy;
                    double z = (start_z + k) * hz;
                    double value = 0.0;
                    
                    switch (data_type) {
                        case NUMERICAL:
                            value = (*numerical)(i, j, k);
                            break;
                        case ANALYTICAL:
                            value = u_analytical(x, y, z, current_time);
                            break;
                        case ERROR:
                            {
                                double exact = u_analytical(x, y, z, current_time);
                                value = fabs((*numerical)(i, j, k) - exact);
                            }
                            break;
                        default:
                            value = 0.0;
                            break;
                    }
                    
                    int idx = i * local_ny * local_nz + j * local_nz + k;
                    data[idx] = value;
                }
            }
        }

        MPI_Send(data.data(), data.size(), MPI_DOUBLE, 0, 4, cart_comm);
    }
}

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);
    
    if (mpi_rank == 0) {
        cout << "MPI Wave Equation" << endl;
        cout << "Grid: " << N << "x" << N << "x" << N << endl;
        cout << "Time steps: " << K << endl;
        cout << "Processes: " << mpi_size << endl;
        cout << "Decomposition: " << mpi_dims[0] << "x" << mpi_dims[1] << "x" << mpi_dims[2] << endl;
    }
    
    double max_lambda = a_squared * tau2 * (1.0/hx2 + 1.0/hy2 + 1.0/hz2);
    if (mpi_rank == 0) {
        cout << "Courant number: " << max_lambda << endl;
        if (max_lambda > 1.0) {
            cout << "Warning: Scheme may be unstable!" << endl;
        }
    }
    
    Grid3D u0(local_nx, local_ny, local_nz);
    Grid3D u1(local_nx, local_ny, local_nz);
    Grid3D u2(local_nx, local_ny, local_nz);
    
    vector<double> neighbor_left(local_ny * local_nz, 0.0);
    vector<double> neighbor_right(local_ny * local_nz, 0.0);
    vector<double> neighbor_bottom(local_nx * local_nz, 0.0);
    vector<double> neighbor_top(local_nx * local_nz, 0.0);
    vector<double> neighbor_back(local_nx * local_ny, 0.0);
    vector<double> neighbor_front(local_nx * local_ny, 0.0);
    
    vector<int> time_steps;
    vector<double> time_values;
    vector<double> max_abs_errors;
    
    double start_time = MPI_Wtime();
    
    initialize_grid(u0);
    
    double max_abs_error, max_rel_error;
    double initial_error = compute_max_error(u0, 0.0, max_abs_error, max_rel_error);
    
    time_steps.push_back(0);
    time_values.push_back(0.0);
    max_abs_errors.push_back(initial_error);
    
    if (mpi_rank == 0) {
        cout << "Step 0, time 0.0, max absolute error = " << initial_error << endl;
    }
    
    exchange_with_neighbors(u0, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);
    compute_first_time_step(u0, u1, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);
    
    double first_step_error = compute_max_error(u1, tau, max_abs_error, max_rel_error);
    
    time_steps.push_back(1);
    time_values.push_back(tau);
    max_abs_errors.push_back(first_step_error);
    
    if (mpi_rank == 0) {
        cout << "Step 1, time " << tau << ", max absolute error = " << first_step_error << endl;
    }
    
    for (int n = 1; n < K; n++) {
        exchange_with_neighbors(u1, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);
        compute_time_step(u0, u1, u2, neighbor_left, neighbor_right, neighbor_bottom, neighbor_top, neighbor_back, neighbor_front);
        
        swap(u0, u1);
        swap(u1, u2);
        
        double current_time = (n + 1) * tau;
        double current_error = compute_max_error(u1, current_time, max_abs_error, max_rel_error);
        
        
        time_steps.push_back(n + 1);
        time_values.push_back(current_time);
        max_abs_errors.push_back(current_error);
        

        if (n % 20 == 0 && mpi_rank == 0) {
            cout << "Step " << (n + 1) << ", time " << current_time 
                 << ", max absolute error = " << current_error << endl;
        }
    }
    
    double end_time = MPI_Wtime();
    
    double final_abs_error = compute_max_error(u1, T, max_abs_error, max_rel_error);
    double final_rel_error = max_rel_error;
    
    if (mpi_rank == 0) {
        cout << "\n=== FINAL RESULTS ===" << endl;
        cout << "Final absolute error: " << final_abs_error << endl;
        cout << "Final relative error: " << final_rel_error << endl;
        cout << "Total time: " << (end_time - start_time) << " seconds" << endl;
        
        write_data_to_file(nullptr, TIME_ERRORS, "time_step_errors.txt", 
                          0.0, &time_steps, &time_values, &max_abs_errors);
    }
    
    //write_data_to_file(&u1, NUMERICAL, "numerical.txt", T);
    //write_data_to_file(&u1, ANALYTICAL, "analytical.txt", T);
    //write_data_to_file(&u1, ERROR, "error.txt", T);
    
    finalize_mpi();
    return 0;
}