
#include <iostream>
using namespace std;

int main()
{
    // int arr[5] = { 1, 2, 3, 4, 5 };
    // int* ptr = arr;

    // cout << "*arr = " << *arr << endl;
    // cout << "arr = " << arr << endl;
    // cout << "&arr = " << &arr << endl;
    // cout << "*ptr = " << *ptr << endl;
    // cout << "ptr = " << ptr << endl;
    // cout << "&ptr = " << &ptr << endl;

    // return 0;

//     int x = 10;       // an integer variable
//     int* p = &x;      // p stores the address of x

// cout << "Value of x: " << x << endl;        // 10
// cout << "Address of x: " << &x << endl;     // e.g. 0x7ffee1234
// cout << "Value of p (address of x): " << p << endl;  
// cout << "Value pointed by p: " << *p << endl;  // dereference p to get 10

    int x = 20;
int* p = &x;      // pointer to x
int** pp = &p;    // pointer to p

cout << "Value of x: " << x << endl;           // 20
cout << "Value pointed by p: " << *p << endl;  // 20
cout << "Value pointed by pp: " << **pp << endl; // 20

cout << "Address of x: " << &x << endl;         // e.g. 0x7ffee1234
cout << "Value of p (address of x): " << p << endl;  // same as above
cout << "Value of pp (address of p): " << pp << endl;


}