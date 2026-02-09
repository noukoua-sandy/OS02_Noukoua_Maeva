#include <cassert>
#include <vector>
#include <iostream>

// Classe représentant une matrice dense
class Matrix : public std::vector<double>
{
public:
    Matrix(int dim);
    Matrix(int nrows, int ncols);

    Matrix(const Matrix& A) = delete;
    Matrix(Matrix&& A) = default;
    ~Matrix() = default;

    Matrix& operator=(const Matrix& A) = delete;
    Matrix& operator=(Matrix&& A) = default;

    // Accès aux coefficients
    double& operator()(int i, int j) {
        return m_arr_coefs[i + j * m_nrows];
    }

    double operator()(int i, int j) const {
        return m_arr_coefs[i + j * m_nrows];
    }

    // Produit matrice-vecteur
    std::vector<double> operator*(const std::vector<double>& u) const;

    // Affichage
    std::ostream& print(std::ostream& out) const
    {
        out << "[\n";
        for (int i = 0; i < m_nrows; ++i) {
            out << " [ ";
            for (int j = 0; j < m_ncols; ++j) {
                out << (*this)(i, j) << " ";
            }
            out << "]\n";
        }
        out << "]";
        return out;
    }

private:
    int m_nrows, m_ncols;
    std::vector<double> m_arr_coefs;
};

// Surcharge de l’opérateur <<
inline std::ostream&
operator<<(std::ostream& out, const Matrix& A)
{
    return A.print(out);
}

// Affichage d’un vecteur
inline std::ostream&
operator<<(std::ostream& out, const std::vector<double>& u)
{
    out << "[ ";
    for (const auto& x : u)
        out << x << " ";
    out << "]";
    return out;
}

// Implémentation du produit matrice-vecteur
std::vector<double>
Matrix::operator*(const std::vector<double>& u) const
{
    assert(u.size() == static_cast<unsigned>(m_ncols));
    std::vector<double> v(m_nrows, 0.0);

    for (int i = 0; i < m_nrows; ++i) {
        for (int j = 0; j < m_ncols; ++j) {
            v[i] += (*this)(i, j) * u[j];
        }
    }
    return v;
}

// Constructeur matrice carrée
Matrix::Matrix(int dim)
    : m_nrows(dim), m_ncols(dim), m_arr_coefs(dim * dim)
{
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            (*this)(i, j) = (i + j) % dim;
        }
    }
}

// Constructeur matrice rectangulaire
Matrix::Matrix(int nrows, int ncols)
    : m_nrows(nrows), m_ncols(ncols), m_arr_coefs(nrows * ncols)
{
    int dim = (nrows > ncols ? nrows : ncols);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            (*this)(i, j) = (i + j) % dim;
        }
    }
}

// Programme principal
int main(int argc, char* argv[])
{
    const int N = 120;

    Matrix A(N);
    std::cout << "A : " << A << std::endl;

    std::vector<double> u(N);
    for (int i = 0; i < N; ++i)
        u[i] = i + 1;

    std::cout << "u : " << u << std::endl;

    std::vector<double> v = A * u;
    std::cout << "A.u = " << v << std::endl;

    return EXIT_SUCCESS;
}
