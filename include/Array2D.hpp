#pragma once

template <typename T> struct Array2D
{
    T *data;
    int nx, ny;

    __host__ __device__ T &operator()(int i, int j)
    {
        return data[i * ny + j];
    }

    __host__ __device__ const T &operator()(int i, int j) const
    {
        return data[i * ny + j];
    }
};

template <typename T, size_t N> struct Array2DSquare
{
    T *data;

    __host__ __device__ T &operator()(int i, int j)
    {
        return data[i * N + j];
    }

    __host__ __device__ const T &operator()(int i, int j) const
    {
        return data[i * N + j];
    }
};
