#ifndef UNREACHABLE_H
#define UNREACHABLE_H

#if defined(_MSC_VER)
    #define UNREACHABLE(msg) __assume(false)
#elif defined(__CUDA_ARCH__)
    #define UNREACHABLE(msg) __trap()
#else
    #define UNREACHABLE(msg) __builtin_unreachable()
#endif

#endif
