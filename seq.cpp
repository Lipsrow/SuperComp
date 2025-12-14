#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <chrono>

using namespace std;
using my_clock_t = chrono::high_resolution_clock;
using second_t = chrono::duration<double, ratio<1>>;

const double Lx = 1;
const double Ly = 1;
const double Lz = 1;
const double T = 0.1;
const int N = 128;
const int K = 100;

const double hx = Lx / (N - 1), hy = Ly / (N - 1), hz = Lz / (N - 1);
const double tau = T / K;
const double hx2 = hx * hx, hy2 = hy * hy, hz2 = hz * hz;
const double tau2 = tau * tau;

const double a_squared = 4.0 / (9.0/(Lx*Lx) + 4.0/(Ly*Ly) + 4.0/(Lz*Lz));

const double a2_hx2 = a_squared / hx2;
const double a2_hy2 = a_squared / hy2;
const double a2_hz2 = a_squared / hz2;

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
    
    int size() const { 
        return nx * ny * nz; 
    }
    int get_nx() const { 
        return nx; 
    }
    int get_ny() const { 
        return ny; 
    }
    int get_nz() const { 
        return nz; 
    }
    
    const vector<double>& get_data() const { 
        return data; 
    }
};

inline double u_analytical(double x, double y, double z, double t) {
    const double factor1 = 3.0 * M_PI / Lx;
    const double factor2 = 2.0 * M_PI / Ly;
    const double factor3 = 2.0 * M_PI / Lz;
    const double at_val = 2.0 * M_PI;
    
    return sin(factor1 * x) * cos(factor2 * y) * cos(factor3 * z) * cos(at_val * t + 4.0 * M_PI);
}

inline double phi(double x, double y, double z) {
    return u_analytical(x, y, z, 0.0);
}

inline double laplacian(const Grid3D& u, int i, int j, int k) {
    int im1 = (i > 0) ? i - 1 : i;
    int ip1 = (i < N - 1) ? i + 1 : i;
    int jm1 = (j > 0) ? j - 1 : N - 1;
    int jp1 = (j < N - 1) ? j + 1 : 0;
    int km1 = (k > 0) ? k - 1 : N - 1;
    int kp1 = (k < N - 1) ? k + 1 : 0;
    
    double dx = (u(im1, j, k) - 2 * u(i, j, k) + u(ip1, j, k)) * a2_hx2;
    double dy = (u(i, jm1, k) - 2 * u(i, j, k) + u(i, jp1, k)) * a2_hy2;
    double dz = (u(i, j, km1) - 2 * u(i, j, k) + u(i, j, kp1)) * a2_hz2;
    
    return dx + dy + dz;
}

void initialize_grid(Grid3D& grid) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * hx;
            double y = j * hy;
            for (int k = 0; k < N; k++) {
                double z = k * hz;
                grid(i, j, k) = phi(x, y, z);
            }
        }
    }
    
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            grid(0, j, k) = 0.0;
            grid(N-1, j, k) = 0.0;
        }
    }
}

void compute_first_time_step_fast(const Grid3D& u0, Grid3D& u1) {
    const double tau2_half = tau2 * 0.5;
    
    for (int i = 1; i < N-1; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                u1(i, j, k) = u0(i, j, k) + tau2_half * laplacian(u0, i, j, k);
            }
        }
    }
    
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            u1(0, j, k) = 0.0;
            u1(N-1, j, k) = 0.0;
        }
    }
}

void compute_time_step_fast(const Grid3D& u0, const Grid3D& u1, Grid3D& u2) {
    for (int i = 1; i < N-1; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                u2(i, j, k) = 2*u1(i, j, k) - u0(i, j, k) + 
                              tau2 * laplacian(u1, i, j, k);
            }
        }
    }
    
    for (int j = 0; j < N; j++) {
        for (int k = 0; k < N; k++) {
            u2(0, j, k) = 0.0;
            u2(N-1, j, k) = 0.0;
        }
    }
}

double compute_max_abs_error_at_time_step(const Grid3D& numerical, double current_time) {
    double max_abs_error = 0.0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * hx;
            double y = j * hy;
            for (int k = 0; k < N; k++) {
                double z = k * hz;
                double exact = u_analytical(x, y, z, current_time);
                double num = numerical(i, j, k);
                double abs_error = fabs(num - exact);
                
                if (abs_error > max_abs_error) {
                    max_abs_error = abs_error;
                }
            }
        }
    }
    
    return max_abs_error;
}

void compute_errors(const Grid3D& numerical, const Grid3D& analytical,
                   double& max_abs_error, double& max_rel_error, 
                   double& rms_error, double& max_analytical_value) {
    
    max_abs_error = 0.0;
    max_rel_error = 0.0;
    rms_error = 0.0;
    max_analytical_value = 0.0;
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double exact = analytical(i, j, k);
                double num = numerical(i, j, k);
                double abs_error = fabs(num - exact);
                double rel_error = 0.0;
                
                if (fabs(exact) > 1e-12) {
                    rel_error = abs_error / fabs(exact);
                }
                
                if (abs_error > max_abs_error) {
                    max_abs_error = abs_error;
                }
                if (rel_error > max_rel_error) {
                    max_rel_error = rel_error;
                }
                rms_error += abs_error * abs_error;
                if (fabs(exact) > max_analytical_value) {
                    max_analytical_value = fabs(exact);
                }
            }
        }
    }
    
    rms_error = sqrt(rms_error / (N * N * N));
}

enum DataType {
    NUMERICAL,
    ANALYTICAL,
    ERROR,
    TIME_ERRORS
};

void write_data_to_file(const Grid3D* numerical, const Grid3D* analytical, 
                       DataType data_type, const string& filename, 
                       double current_time = 0.0,
                       const vector<int>* time_steps = nullptr,
                       const vector<double>* time_values = nullptr,
                       const vector<double>* errors = nullptr) {
    
    if (data_type == TIME_ERRORS) {
        ofstream file(filename);
        if (!file.is_open()) {
            cerr << "Не удалось открыть файл для записи погрешностей по шагам!" << endl;
            return;
        }
        
        file << "Time_Step Time Max_Absolute_Error" << endl;
        if (time_steps && time_values && errors) {
            for (size_t i = 0; i < time_steps->size(); i++) {
                file << (*time_steps)[i] << " " << (*time_values)[i] << " " << (*errors)[i] << endl;
            }
        }
        file.close();
        return;
    }
    
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
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                double value = 0.0;
                
                switch (data_type) {
                    case NUMERICAL:
                        if (numerical) {
                            value = (*numerical)(i, j, k);
                        }
                        break;
                    case ANALYTICAL:
                        {
                            double x = i * hx;
                            double y = j * hy;
                            double z = k * hz;
                            value = u_analytical(x, y, z, current_time);
                        }
                        break;
                    case ERROR:
                        if (numerical && analytical) {
                            value = fabs((*numerical)(i, j, k) - (*analytical)(i, j, k));
                        } else if (numerical) {
                            double x = i * hx;
                            double y = j * hy;
                            double z = k * hz;
                            double exact = u_analytical(x, y, z, current_time);
                            value = fabs((*numerical)(i, j, k) - exact);
                        }
                        break;
                    default:
                        value = 0.0;
                        break;
                }
                
                file << i << " " << j << " " << k << " " << value << endl;
            }
        }
    }
    
    file.close();
}

void write_time_step_errors_to_file(const vector<int>& steps, const vector<double>& times, 
                                   const vector<double>& errors, const string& filename) {
    write_data_to_file(nullptr, nullptr, TIME_ERRORS, filename, 0.0, &steps, &times, &errors);
}

int main() {
    auto start_time = my_clock_t::now();
    cout << "Grid size: " << N << "x" << N << "x" << N << endl;
    cout << "Time steps: " << K << endl;
    cout << "LX: " << Lx << endl;
    
    double max_lambda = a_squared * tau2 * (1.0/hx2 + 1.0/hy2 + 1.0/hz2);
    cout << "Courant number: " << max_lambda << endl;
    
    if (max_lambda > 1.0) {
        cout << "Схема не устойчива!" << endl;
        return 1;
    }

    Grid3D u0(N, N, N), u1(N, N, N), u2(N, N, N);
    
    vector<int> time_steps;
    vector<double> time_values;
    vector<double> max_abs_errors;
    
    initialize_grid(u0);
    
    double initial_error = compute_max_abs_error_at_time_step(u0, 0.0);
    time_steps.push_back(0);
    time_values.push_back(0.0);
    max_abs_errors.push_back(initial_error);
    
    cout << "Step 0/" << K << ", time = 0.0, max error = " << initial_error << endl;
    
    compute_first_time_step_fast(u0, u1);
    
    double first_step_error = compute_max_abs_error_at_time_step(u1, tau);
    time_steps.push_back(1);
    time_values.push_back(tau);
    max_abs_errors.push_back(first_step_error);
    
    cout << "Step 1/" << K << ", time = " << tau << ", max error = " << first_step_error << endl;
    
    for (int n = 1; n < K; n++) {
        compute_time_step_fast(u0, u1, u2);
        
        swap(u0, u1);
        swap(u1, u2);
        
        double current_time = (n + 1) * tau;
        double current_error = compute_max_abs_error_at_time_step(u1, current_time);
        
        time_steps.push_back(n + 1);
        time_values.push_back(current_time);
        max_abs_errors.push_back(current_error);
        
        if (n % 20 == 0) {
            cout << "Step " << (n + 1) << "/" << K << ", time = " << current_time 
                 << ", max error = " << current_error << endl;
        }
    }
    
    write_time_step_errors_to_file(time_steps, time_values, max_abs_errors, "time_step_errors.txt");
    
    Grid3D u_analytical_grid(N, N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * hx;
            double y = j * hy;
            for (int k = 0; k < N; k++) {
                double z = k * hz;
                u_analytical_grid(i, j, k) = u_analytical(x, y, z, T);
            }
        }
    }
    
    double max_abs_error, max_rel_error, rms_error, max_analytical_value;
    compute_errors(u1, u_analytical_grid, max_abs_error, max_rel_error, rms_error, max_analytical_value);
    
    cout << "\nRESULTS" << endl;
    cout << "Maximum absolute error: " << max_abs_error << endl;
    cout << "Maximum relative error: " << max_rel_error << endl;
    cout << "RMS error: " << rms_error << endl;
    cout << "Time step errors saved to: time_step_errors.txt" << endl;
    auto end_time = my_clock_t::now();
    chrono::duration<double> elapsed = end_time - start_time;
    
    cout << "\nPROGRAM EXECUTION TIME: " << elapsed.count() << " seconds" << endl;
    
    //write_data_to_file(&u1, nullptr, NUMERICAL, "numerical.txt", T);
    //write_data_to_file(nullptr, nullptr, ANALYTICAL, "analytical.txt", T);
    //write_data_to_file(&u1, &u_analytical_grid, ERROR, "error.txt", T);
    
    return 0;
}
