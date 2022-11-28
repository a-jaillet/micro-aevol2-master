#include <iostream>
#include "BitSet.h"

using namespace std;


int main()
{
    cout << "hello" << endl;
    int size = 40;
    BitSet dna;
    dna.init(size);
    dna.print();

    for(int i = 0; i < size; i ++)
    {
        dna.set(i);
        cout << dna.get(i) << endl;
        dna.print();
    }
    dna.print();
}