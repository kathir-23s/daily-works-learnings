#include <iostream>
#include <cstdint>

int8_t add(int8_t a,int8_t b){ return a+b; }
int16_t add(int16_t a,int16_t b){ return a+b; }
float add(float a,float b){ return a+b; }
double add(double a,double b){ return a+b; }
bool add(bool a,bool b){ return a+b; }

template <typename T>
T addT (T a, T b){ return a+b; }

template<>
bool addT(bool a, bool b){ return a || b;}

int main() {
    // call overloads
    auto a = add((int8_t)1, (int8_t)2);
    auto b = add((int16_t)10, (int16_t)20);
    auto c = add(1.5f, 2.5f);
    auto d = add(1.5, 2.5);
    auto e = add(true, false);

    // call templates
    auto ta = addT<int8_t>(3, 4);
    auto tb = addT<int16_t>(30, 40);
    auto tc = addT<float>(2.3f, 1.1f);
    auto td = addT<double>(2.5, 1.2);
    auto te = addT<bool>(true, true);

    std::cout << (int)a << " " << b << " " << c << " " << d << " " << e << "\n";
    std::cout << (int)ta << " " << tb << " " << tc << " " << td << " " << te << "\n";
}

