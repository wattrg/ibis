#ifndef FIELD_H
#define FIELD_H

struct Field {
public:
    Field(int n);
    ~Field();

    inline int length() {return _length;}
    inline double& operator [] (int index) {return _data[index];}

private:
    int _length;
    double *_data;
};

#endif
