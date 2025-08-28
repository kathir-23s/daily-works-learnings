#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <numeric>  // for accumulate
#include <cmath>    // for pow

class TensorT {
public:
    std::vector<std::vector<double>> data;
    std::pair<int, int> shape;

    // Constructor
    TensorT(const std::vector<std::vector<double>>& input) {
        if (input.empty() || input[0].empty()) {
            throw std::invalid_argument("Tensor cannot be empty");
        }
        // Check rectangular
        size_t row_len = input[0].size();
        for (size_t i = 0; i < input.size(); i++) {
            if (input[i].size() != row_len) {
                throw std::invalid_argument("Ragged tensor not allowed");
            }
        }
        data = input;
        shape = { (int)input.size(), (int)input[0].size() };
    }

    // Overload printing
    void print() const {
        std::cout << "tensor:\n";
        for (const auto& row : data) {
            std::cout << "[";
            for (size_t j = 0; j < row.size(); j++) {
                std::cout << row[j];
                if (j < row.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        std::cout << "shape: (" << shape.first << ", " << shape.second << ")\n";
    }

    // -------- FACTORIES --------
    static TensorT unit_tensor(int unit, int rows, int cols) {
        if (unit != 0 && unit != 1)
            throw std::invalid_argument("unit must be 0 or 1");
        return TensorT(
            std::vector<std::vector<double>>(rows, std::vector<double>(cols, (double)unit))
        );
    }

    static TensorT random_tensor(int rows, int cols) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> d(0.0, 1.0);
        std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                out[i][j] = d(gen);
        return TensorT(out);
    }

    // -------- BASIC OPS --------
    TensorT operator+(const TensorT& other) const {
        if (shape != other.shape)
            throw std::invalid_argument("Shape mismatch in +");
        std::vector<std::vector<double>> result(shape.first, std::vector<double>(shape.second));
        for (int i = 0; i < shape.first; i++)
            for (int j = 0; j < shape.second; j++)
                result[i][j] = data[i][j] + other.data[i][j];
        return TensorT(result);
    }

    TensorT operator-(const TensorT& other) const {
        if (shape != other.shape)
            throw std::invalid_argument("Shape mismatch in -");
        std::vector<std::vector<double>> result(shape.first, std::vector<double>(shape.second));
        for (int i = 0; i < shape.first; i++)
            for (int j = 0; j < shape.second; j++)
                result[i][j] = data[i][j] - other.data[i][j];
        return TensorT(result);
    }

    TensorT operator*(const TensorT& other) const {
        if (shape != other.shape)
            throw std::invalid_argument("Shape mismatch in *");
        std::vector<std::vector<double>> result(shape.first, std::vector<double>(shape.second));
        for (int i = 0; i < shape.first; i++)
            for (int j = 0; j < shape.second; j++)
                result[i][j] = data[i][j] * other.data[i][j];
        return TensorT(result);
    }

    TensorT operator-() const { // Negation
        std::vector<std::vector<double>> result(shape.first, std::vector<double>(shape.second));
        for (int i = 0; i < shape.first; i++)
            for (int j = 0; j < shape.second; j++)
                result[i][j] = -data[i][j];
        return TensorT(result);
    }

    TensorT pow(double exp) const {
        std::vector<std::vector<double>> result(shape.first, std::vector<double>(shape.second));
        for (int i = 0; i < shape.first; i++)
            for (int j = 0; j < shape.second; j++)
                result[i][j] = std::pow(data[i][j], exp);
        return TensorT(result);
    }

    // -------- MATRIX OPS --------
    TensorT matmul(const TensorT& other) const {
        if (shape.second != other.shape.first)
            throw std::invalid_argument("Incompatible matmul dims");
        std::vector<std::vector<double>> result(shape.first, std::vector<double>(other.shape.second, 0.0));
        for (int i = 0; i < shape.first; i++) {
            for (int j = 0; j < other.shape.second; j++) {
                for (int k = 0; k < shape.second; k++) {
                    result[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return TensorT(result);
    }

    TensorT transpose() const {
        std::vector<std::vector<double>> result(shape.second, std::vector<double>(shape.first));
        for (int i = 0; i < shape.first; i++)
            for (int j = 0; j < shape.second; j++)
                result[j][i] = data[i][j];
        return TensorT(result);
    }

    std::vector<double> flatten() const {
        std::vector<double> flat;
        flat.reserve(shape.first * shape.second);
        for (const auto& row : data)
            for (double v : row) flat.push_back(v);
        return flat;
    }

    TensorT reshape(int new_rows, int new_cols) const {
        if (shape.first * shape.second != new_rows * new_cols)
            throw std::invalid_argument("Incompatible reshape size");
        std::vector<double> flat = flatten();
        std::vector<std::vector<double>> reshaped(new_rows, std::vector<double>(new_cols));
        int idx = 0;
        for (int i = 0; i < new_rows; i++)
            for (int j = 0; j < new_cols; j++)
                reshaped[i][j] = flat[idx++];
        return TensorT(reshaped);
    }
};

// ---------------- MAIN TEST ----------------
int main() {
    TensorT A = TensorT::random_tensor(2, 3);
    TensorT B = TensorT::unit_tensor(1, 2, 3);

    std::cout << "Matrix A:\n"; A.print();
    std::cout << "Matrix B:\n"; B.print();

    TensorT C = A + B;
    std::cout << "A + B:\n"; C.print();

    TensorT D = A.transpose();
    std::cout << "Transpose(A):\n"; D.print();

    TensorT E = A.reshape(3, 2);
    std::cout << "Reshaped (3x2):\n"; E.print();

    TensorT F = A.matmul(D);
    std::cout << "A x A^T:\n"; F.print();

    return 0;
}
