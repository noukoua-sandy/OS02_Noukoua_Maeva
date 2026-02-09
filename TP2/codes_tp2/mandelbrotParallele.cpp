#include <iostream>
#include <cstdlib>
#include <string>
#include <chrono>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
#include <mpi.h>

struct Complex
{
    Complex() : real(0.), imag(0.) {}
    Complex(double r, double i) : real(r), imag(i) {}

    Complex operator + (const Complex& z) { return Complex(real + z.real, imag + z.imag); }
    Complex operator * (const Complex& z) { return Complex(real*z.real - imag*z.imag, real*z.imag + imag*z.real); }

    double sqNorm() { return real*real + imag*imag; }

    double real, imag;
};

std::ostream& operator << (std::ostream& out, const Complex& c)
{
    out << "(" << c.real << "," << c.imag << ")" << std::endl;
    return out;
}

void savePicture(const std::string& filename, int W, int H,
                 const std::vector<int>& nbIters, int maxIter)
{
    double scaleCol = 1.0 / maxIter;
    std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);

    ofs << "P6\n" << W << " " << H << "\n255\n";

    for (int i = 0; i < W * H; ++i) {
        double iter = scaleCol * nbIters[i];
        unsigned char r = (unsigned char)(256 - (unsigned(iter * 256.) & 0xFF));
        unsigned char b = (unsigned char)(256 - (unsigned(iter * 65536) & 0xFF));
        unsigned char g = (unsigned char)(256 - (unsigned(iter * 16777216) & 0xFF));
        ofs << r << g << b;
    }
    ofs.close();
}

int iterMandelbrot(int maxIter, const Complex& c)
{
    Complex z{0., 0.};

    if (c.real*c.real + c.imag*c.imag < 0.0625)
        return maxIter;
    if ((c.real + 1)*(c.real + 1) + c.imag*c.imag < 0.0625)
        return maxIter;

    if ((c.real > -0.75) && (c.real < 0.5)) {
        Complex ct{c.real - 0.25, c.imag};
        double ctnrm2 = sqrt(ct.sqNorm());
        if (ctnrm2 < 0.5*(1 - ct.real/ctnrm2))
            return maxIter;
    }

    int niter = 0;
    while ((z.sqNorm() < 4.) && (niter < maxIter)) {
        z = z*z + c;
        ++niter;
    }
    return niter;
}

void computeMandelbrotSetRow(int W, int H, int maxIter, int num_ligne, int* pixels)
{
    double scaleX = 3.0 / (W - 1);
    double scaleY = 2.25 / (H - 1.0);

    for (int j = 0; j < W; ++j) {
        Complex c{-2.0 + j*scaleX, -1.125 + num_ligne*scaleY};
        pixels[j] = iterMandelbrot(maxIter, c);
    }
}

std::vector<int> computeMandelbrotSet(int W, int H, int maxIter)
{
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    std::vector<int> pixel(W * H);

    for (int i = 0; i < H; ++i) {
        computeMandelbrotSetRow(W, H, maxIter, i, pixel.data() + W*(H - i - 1));
    }

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "mandelbrot : "
              << std::fixed
              << std::setprecision(6)
              << std::setw(5) << elapsed_seconds.count() << " s"
              << std::endl;

    return pixel;
}

int main(int argc, char* argv[])
{
    const int W = 800;
    const int H = 600;
    const int maxIter = 8 * 65536;

    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int MASTER = 0;

    // Hypothèse simple : W divisible par world_size
    int localW = W / world_size;
    int data_size = localW * H;

    auto iters = computeMandelbrotSet(localW, H, maxIter);

    int* results = nullptr;
    if (rank == MASTER)
        results = new int[W * H];

    MPI_Gather(iters.data(), data_size, MPI_INT,
               results, data_size, MPI_INT,
               MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        std::vector<int> full(results, results + W * H);
        savePicture("mandelbrot_parallel.tga", W, H, full, maxIter);
        delete[] results;
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
