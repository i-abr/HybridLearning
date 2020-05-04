#include <iostream>

void test(int &);

int main() {
    int t = 89;
    std::cout << t << std::endl;
    test(t);
    std::cout << t << std::endl;
    return 0;
}


void test(int & fh) {
    fh = 0;
    std::cout << "sdfsd " << fh << std::endl; 
}
