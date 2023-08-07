#ifndef GAS_H
#define GAS_H

struct GasStates {
public:
    ~GasStates();
    GasStates(int n);

    inline double *pressure(){return _pressure;}
    inline double *tempe(){return _temp;}
    
private:
    int _size;
    double *_pressure;
    double *_temp;
};

#endif
