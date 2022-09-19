// -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; -*-
//
// Copyright (c) 2008, Casey Duncan (casey dot duncan at gmail dot com)
// Copyright (c) 2017, Zev Benjamin <zev@strangersgate.com>
// see LICENSE.txt for details

#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <float.h>
#include "_noise.h"

typedef struct _NoiseArgs NoiseArgs;

typedef PyObject* (*ScalarFunc)(NoiseArgs* args);
typedef float (*DispatchFunc)(NoiseArgs* args, float **coord);

typedef struct _NoiseArgs {
    int ndims;
    int nops;
    //  dim_vals is an array of the form [xs, ys, ...]
    PyObject **dim_vals;
    PyArrayObject **op;
    npy_uint32 *op_flags;
    PyArray_Descr **op_dtypes;

    ScalarFunc scalar_func;
    DispatchFunc dispatch_func;

    int octaves;
    float persistence;
    float lacunarity;

    // Only used for noise2
    float repeatx;
    float repeaty;
    float z;
} NoiseArgs;

// 2D simplex skew factors
#define F2 0.3660254037844386f  // 0.5 * (sqrt(3.0) - 1.0)
#define G2 0.21132486540518713f // (3.0 - sqrt(3.0)) / 6.0

float
noise2(float x, float y)
{
    int i1, j1, I, J, c;
    float s = (x + y) * F2;
    float i = floorf(x + s);
    float j = floorf(y + s);
    float t = (i + j) * G2;

    float xx[3], yy[3], f[3];
    float noise[3] = {0.0f, 0.0f, 0.0f};
    int g[3];

    xx[0] = x - (i - t);
    yy[0] = y - (j - t);

    i1 = xx[0] > yy[0];
    j1 = xx[0] <= yy[0];

    xx[2] = xx[0] + G2 * 2.0f - 1.0f;
    yy[2] = yy[0] + G2 * 2.0f - 1.0f;
    xx[1] = xx[0] - i1 + G2;
    yy[1] = yy[0] - j1 + G2;

    I = (int) i & 255;
    J = (int) j & 255;
    g[0] = PERM[I + PERM[J]] % 12;
    g[1] = PERM[I + i1 + PERM[J + j1]] % 12;
    g[2] = PERM[I + 1 + PERM[J + 1]] % 12;

    for (c = 0; c <= 2; c++)
        f[c] = 0.5f - xx[c]*xx[c] - yy[c]*yy[c];

    for (c = 0; c <= 2; c++)
        if (f[c] > 0)
            noise[c] = f[c]*f[c]*f[c]*f[c] * (GRAD3[g[c]][0]*xx[c] + GRAD3[g[c]][1]*yy[c]);

    return (noise[0] + noise[1] + noise[2]) * 70.0f;
}

#define dot3(v1, v2) ((v1)[0]*(v2)[0] + (v1)[1]*(v2)[1] + (v1)[2]*(v2)[2])

#define ASSIGN(a, v0, v1, v2) (a)[0] = v0; (a)[1] = v1; (a)[2] = v2;

#define F3 (1.0f / 3.0f)
#define G3 (1.0f / 6.0f)

float
noise3(float x, float y, float z)
{
    int c, o1[3], o2[3], g[4], I, J, K;
    float f[4], noise[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float s = (x + y + z) * F3;
    float i = floorf(x + s);
    float j = floorf(y + s);
    float k = floorf(z + s);
    float t = (i + j + k) * G3;

    float pos[4][3];

    pos[0][0] = x - (i - t);
    pos[0][1] = y - (j - t);
    pos[0][2] = z - (k - t);

    if (pos[0][0] >= pos[0][1]) {
        if (pos[0][1] >= pos[0][2]) {
            ASSIGN(o1, 1, 0, 0);
            ASSIGN(o2, 1, 1, 0);
        } else if (pos[0][0] >= pos[0][2]) {
            ASSIGN(o1, 1, 0, 0);
            ASSIGN(o2, 1, 0, 1);
        } else {
            ASSIGN(o1, 0, 0, 1);
            ASSIGN(o2, 1, 0, 1);
        }
    } else {
        if (pos[0][1] < pos[0][2]) {
            ASSIGN(o1, 0, 0, 1);
            ASSIGN(o2, 0, 1, 1);
        } else if (pos[0][0] < pos[0][2]) {
            ASSIGN(o1, 0, 1, 0);
            ASSIGN(o2, 0, 1, 1);
        } else {
            ASSIGN(o1, 0, 1, 0);
            ASSIGN(o2, 1, 1, 0);
        }
    }

    for (c = 0; c <= 2; c++) {
        pos[3][c] = pos[0][c] - 1.0f + 3.0f * G3;
        pos[2][c] = pos[0][c] - o2[c] + 2.0f * G3;
        pos[1][c] = pos[0][c] - o1[c] + G3;
    }

    I = (int) i & 255;
    J = (int) j & 255;
    K = (int) k & 255;
    g[0] = PERM[I + PERM[J + PERM[K]]] % 12;
    g[1] = PERM[I + o1[0] + PERM[J + o1[1] + PERM[o1[2] + K]]] % 12;
    g[2] = PERM[I + o2[0] + PERM[J + o2[1] + PERM[o2[2] + K]]] % 12;
    g[3] = PERM[I + 1 + PERM[J + 1 + PERM[K + 1]]] % 12;

    for (c = 0; c <= 3; c++) {
        f[c] = 0.6f - pos[c][0]*pos[c][0] - pos[c][1]*pos[c][1] - pos[c][2]*pos[c][2];
    }

    for (c = 0; c <= 3; c++) {
        if (f[c] > 0) {
            noise[c] = f[c]*f[c]*f[c]*f[c] * dot3(pos[c], GRAD3[g[c]]);
        }
    }

    return (noise[0] + noise[1] + noise[2] + noise[3]) * 32.0f;
}

inline float
fbm_noise3(float x, float y, float z, int octaves, float persistence, float lacunarity) {
    float freq = 1.0f;
    float amp = 1.0f;
    float max = 1.0f;
    float total = noise3(x, y, z);
    int i;

    for (i = 1; i < octaves; ++i) {
        freq *= lacunarity;
        amp *= persistence;
        max += amp;
        total += noise3(x * freq, y * freq, z * freq) * amp;
    }
    return total / max;
}

#define dot4(v1, x, y, z, w) ((v1)[0]*(x) + (v1)[1]*(y) + (v1)[2]*(z) + (v1)[3]*(w))

#define F4 0.30901699437494745f /* (sqrt(5.0) - 1.0) / 4.0 */
#define G4 0.1381966011250105f /* (5.0 - sqrt(5.0)) / 20.0 */

float
noise4(float x, float y, float z, float w) {
    float noise[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float s = (x + y + z + w) * F4;
    float i = floorf(x + s);
    float j = floorf(y + s);
    float k = floorf(z + s);
    float l = floorf(w + s);
    float t = (i + j + k + l) * G4;

    float x0 = x - (i - t);
    float y0 = y - (j - t);
    float z0 = z - (k - t);
    float w0 = w - (l - t);

    int c = (x0 > y0)*32 + (x0 > z0)*16 + (y0 > z0)*8 + (x0 > w0)*4 + (y0 > w0)*2 + (z0 > w0);
    int i1 = SIMPLEX[c][0]>=3;
    int j1 = SIMPLEX[c][1]>=3;
    int k1 = SIMPLEX[c][2]>=3;
    int l1 = SIMPLEX[c][3]>=3;
    int i2 = SIMPLEX[c][0]>=2;
    int j2 = SIMPLEX[c][1]>=2;
    int k2 = SIMPLEX[c][2]>=2;
    int l2 = SIMPLEX[c][3]>=2;
    int i3 = SIMPLEX[c][0]>=1;
    int j3 = SIMPLEX[c][1]>=1;
    int k3 = SIMPLEX[c][2]>=1;
    int l3 = SIMPLEX[c][3]>=1;

    float x1 = x0 - i1 + G4;
    float y1 = y0 - j1 + G4;
    float z1 = z0 - k1 + G4;
    float w1 = w0 - l1 + G4;
    float x2 = x0 - i2 + 2.0f*G4;
    float y2 = y0 - j2 + 2.0f*G4;
    float z2 = z0 - k2 + 2.0f*G4;
    float w2 = w0 - l2 + 2.0f*G4;
    float x3 = x0 - i3 + 3.0f*G4;
    float y3 = y0 - j3 + 3.0f*G4;
    float z3 = z0 - k3 + 3.0f*G4;
    float w3 = w0 - l3 + 3.0f*G4;
    float x4 = x0 - 1.0f + 4.0f*G4;
    float y4 = y0 - 1.0f + 4.0f*G4;
    float z4 = z0 - 1.0f + 4.0f*G4;
    float w4 = w0 - 1.0f + 4.0f*G4;

    int I = (int)i & 255;
    int J = (int)j & 255;
    int K = (int)k & 255;
    int L = (int)l & 255;
    int gi0 = PERM[I + PERM[J + PERM[K + PERM[L]]]] & 0x1f;
    int gi1 = PERM[I + i1 + PERM[J + j1 + PERM[K + k1 + PERM[L + l1]]]] & 0x1f;
    int gi2 = PERM[I + i2 + PERM[J + j2 + PERM[K + k2 + PERM[L + l2]]]] & 0x1f;
    int gi3 = PERM[I + i3 + PERM[J + j3 + PERM[K + k3 + PERM[L + l3]]]] & 0x1f;
    int gi4 = PERM[I + 1 + PERM[J + 1 + PERM[K + 1 + PERM[L + 1]]]] & 0x1f;
    float t0, t1, t2, t3, t4;

    t0 = 0.6f - x0*x0 - y0*y0 - z0*z0 - w0*w0;
    if (t0 >= 0.0f) {
        t0 *= t0;
        noise[0] = t0 * t0 * dot4(GRAD4[gi0], x0, y0, z0, w0);
    }
    t1 = 0.6f - x1*x1 - y1*y1 - z1*z1 - w1*w1;
    if (t1 >= 0.0f) {
        t1 *= t1;
        noise[1] = t1 * t1 * dot4(GRAD4[gi1], x1, y1, z1, w1);
    }
    t2 = 0.6f - x2*x2 - y2*y2 - z2*z2 - w2*w2;
    if (t2 >= 0.0f) {
        t2 *= t2;
        noise[2] = t2 * t2 * dot4(GRAD4[gi2], x2, y2, z2, w2);
    }
    t3 = 0.6f - x3*x3 - y3*y3 - z3*z3 - w3*w3;
    if (t3 >= 0.0f) {
        t3 *= t3;
        noise[3] = t3 * t3 * dot4(GRAD4[gi3], x3, y3, z3, w3);
    }
    t4 = 0.6f - x4*x4 - y4*y4 - z4*z4 - w4*w4;
    if (t4 >= 0.0f) {
        t4 *= t4;
        noise[4] = t4 * t4 * dot4(GRAD4[gi4], x4, y4, z4, w4);
    }

    return 27.0 * (noise[0] + noise[1] + noise[2] + noise[3] + noise[4]);
}

inline float
fbm_noise4(float x, float y, float z, float w, int octaves, float persistence, float lacunarity) {
    float freq = 1.0f;
    float amp = 1.0f;
    float max = 1.0f;
    float total = noise4(x, y, z, w);
    int i;

    for (i = 1; i < octaves; ++i) {
        freq *= lacunarity;
        amp *= persistence;
        max += amp;
        total += noise4(x * freq, y * freq, z * freq, w * freq) * amp;
    }
    return total / max;
}

static float
dispatch_noise2(float x, float y, int octaves, float persistence,
                float lacunarity, float repeatx, float repeaty, float z)
{
    if (repeatx == FLT_MAX && repeaty == FLT_MAX) {
        // Flat noise, no tiling
        float freq = 1.0f;
        float amp = 1.0f;
        float max = 1.0f;
        float total = noise2(x + z, y + z);
        int i;

        for (i = 1; i < octaves; i++) {
            freq *= lacunarity;
            amp *= persistence;
            max += amp;
            total += noise2(x * freq + z, y * freq + z) * amp;
        }
        return total / max;
    } else { // Tiled noise
        float w = z;
        if (repeaty != FLT_MAX) {
            float yf = y * 2.0 / repeaty;
            float yr = repeaty * M_1_PI * 0.5;
            float vy = fast_sin(yf);
            float vyz = fast_cos(yf);
            y = vy * yr;
            w += vyz * yr;
            if (repeatx == FLT_MAX) {
                return fbm_noise3(x, y, w, octaves, persistence, lacunarity);
            }
        }
        if (repeatx != FLT_MAX) {
            float xf = x * 2.0 / repeatx;
            float xr = repeatx * M_1_PI * 0.5;
            float vx = fast_sin(xf);
            float vxz = fast_cos(xf);
            x = vx * xr;
            z += vxz * xr;
            if (repeaty == FLT_MAX) {
                return fbm_noise3(x, y, z, octaves, persistence, lacunarity);
            }
        }
        return fbm_noise4(x, y, z, w, octaves, persistence, lacunarity);
    }
}

static float
dispatch_noise3(float x, float y, float z, int octaves, float persistence,
                float lacunarity)
{
    if (octaves == 1) {
        // Single octave, return simple noise
        return noise3(x, y, z);
    } else {
        // octaves > 1, since we already checked for <= 0
        return fbm_noise3(x, y, z, octaves, persistence, lacunarity);
    }
}

static float
dispatch_noise4(float x, float y, float z, float w, int octaves,
                float persistence, float lacunarity)
{
    if (octaves == 1) {
        // Single octave, return simple noise
        return noise4(x, y, z, w);
    } else {
        // octaves > 1, since we already checked for <= 0
        return fbm_noise4(x, y, z, w, octaves, persistence, lacunarity);
    }
}

static float
dispatch_noise2_args(NoiseArgs *args, float **coord)
{
    return dispatch_noise2(*coord[0],
                           *coord[1],
                           args->octaves, args->persistence, args->lacunarity,
                           args->repeatx, args->repeaty, args->z);
}

static float
dispatch_noise3_args(NoiseArgs *args, float **coord)
{
    return dispatch_noise3(*coord[0],
                           *coord[1],
                           *coord[2],
                           args->octaves, args->persistence, args->lacunarity);
}

static float
dispatch_noise4_args(NoiseArgs *args, float **coord)
{
    return dispatch_noise4(*coord[0],
                           *coord[1],
                           *coord[2],
                           *coord[3],
                           args->octaves, args->persistence, args->lacunarity);
}

static PyObject *
noise2_scalar(NoiseArgs *args)
{
    PyObject* fx = NULL;
    PyObject* fy = NULL;
    PyObject* result = NULL;
    float fresult;

    fx = PyNumber_Float(args->dim_vals[0]);
    if (fx == NULL)
        goto fail_x;

    fy = PyNumber_Float(args->dim_vals[1]);
    if (fy == NULL)
        goto fail_y;

    fresult = dispatch_noise2((float) PyFloat_AsDouble(fx),
                              (float) PyFloat_AsDouble(fy),
                              args->octaves, args->persistence,
                              args->lacunarity,
                              args->repeatx, args->repeaty, args->z);
    result = (PyObject*) PyFloat_FromDouble(fresult);

    Py_DECREF(fy);
fail_y:
    Py_DECREF(fx);
fail_x:
    return result;
}

static PyObject *
noise3_scalar(NoiseArgs *args)
{
    PyObject* fx = NULL;
    PyObject* fy = NULL;
    PyObject* fz = NULL;
    PyObject* result = NULL;
    float fresult;

    fx = PyNumber_Float(args->dim_vals[0]);
    if (fx == NULL)
        goto fail_x;

    fy = PyNumber_Float(args->dim_vals[1]);
    if (fy == NULL)
        goto fail_y;

    fz = PyNumber_Float(args->dim_vals[2]);
    if (fz == NULL)
        goto fail_z;

    fresult = dispatch_noise3((float) PyFloat_AsDouble(fx),
                              (float) PyFloat_AsDouble(fy),
                              (float) PyFloat_AsDouble(fz),
                              args->octaves, args->persistence,
                              args->lacunarity);
    result = (PyObject*) PyFloat_FromDouble(fresult);

    Py_DECREF(fz);
fail_z:
    Py_DECREF(fy);
fail_y:
    Py_DECREF(fx);
fail_x:
    return result;
}

static PyObject *
noise4_scalar(NoiseArgs *args)
{
    PyObject* fx = NULL;
    PyObject* fy = NULL;
    PyObject* fz = NULL;
    PyObject* fw = NULL;
    PyObject* result = NULL;
    float fresult;

    fx = PyNumber_Float(args->dim_vals[0]);
    if (fx == NULL)
        goto fail_x;

    fy = PyNumber_Float(args->dim_vals[1]);
    if (fy == NULL)
        goto fail_y;

    fz = PyNumber_Float(args->dim_vals[2]);
    if (fz == NULL)
        goto fail_z;

    fw = PyNumber_Float(args->dim_vals[3]);
    if (fw == NULL)
        goto fail_w;

    fresult = dispatch_noise4((float) PyFloat_AsDouble(fx),
                              (float) PyFloat_AsDouble(fy),
                              (float) PyFloat_AsDouble(fz),
                              (float) PyFloat_AsDouble(fw),
                              args->octaves, args->persistence,
                              args->lacunarity);
    result = (PyObject*) PyFloat_FromDouble(fresult);

    Py_DECREF(fw);
fail_w:
    Py_DECREF(fz);
fail_z:
    Py_DECREF(fy);
fail_y:
    Py_DECREF(fx);
fail_x:
    return result;
}

static inline PyObject *
py_noise_common(NoiseArgs* args)
{
    static char *var_names[4] = {"xs", "ys", "zs", "ws"};

    int i;
    int all_scalars = 1;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    PyArrayObject *ret = NULL;
    PyArray_Descr *float_type = NULL;
    npy_intp *sizeptr, *strides;
    char **dataptrarray;

    if (args->octaves <= 0) {
        PyErr_SetString(PyExc_ValueError, "Expected octaves value > 0");
        return NULL;
    }

    for (i = 0; i < args->ndims; i++) {
        all_scalars &= PyArray_IsPythonScalar(args->dim_vals[i]);
    }
    if (all_scalars)
        return args->scalar_func(args);

    float_type = PyArray_DescrFromType(NPY_FLOAT);

    for (i = 0; i < args->ndims; i++) {
        args->op[i] = (PyArrayObject*) PyArray_FROM_O(args->dim_vals[i]);
        if (args->op[i] == NULL) {
            PyErr_Format(PyExc_ValueError,
                         "Could not convert argument `%s` to an array of floats",
                         var_names[i]);
            goto fail;
        }
        args->op_flags[i] = NPY_ITER_READONLY;
        args->op_dtypes[i] = float_type;
    }
    args->op[args->nops - 1] = NULL;
    args->op_flags[args->nops - 1] = NPY_ITER_WRITEONLY | NPY_ITER_ALLOCATE;
    args->op_dtypes[args->nops - 1] = float_type;

    iter = NpyIter_MultiNew(args->nops, args->op,
                            NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED,
                            NPY_KEEPORDER, NPY_SAME_KIND_CASTING,
                            args->op_flags, args->op_dtypes);

    if (iter == NULL) {
        goto fail;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    strides = NpyIter_GetInnerStrideArray(iter);
    sizeptr = NpyIter_GetInnerLoopSizePtr(iter);
    dataptrarray = NpyIter_GetDataPtrArray(iter);

    do {
        npy_intp size = *sizeptr;
        float *result;
        int iop;

        while (size--) {
            result = (float*) dataptrarray[args->nops - 1];
            *result = args->dispatch_func(args, (float**) dataptrarray);
            for (iop = 0; iop < args->nops; ++iop) {
                dataptrarray[iop] += strides[iop];
            }
        }
    } while (iternext(iter));

    ret = NpyIter_GetOperandArray(iter)[args->nops - 1];
    Py_INCREF(ret);

    if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
        goto fail;
    }

    Py_DECREF(float_type);
    // Don't deallocate the output array
    for (i = 0; i < args->ndims; i++) {
        Py_DECREF(args->op[i]);
    }

    return PyArray_Return(ret);

fail:
    Py_XDECREF(float_type);
    // Do deallocate the output array
    for (i = 0; i < args->nops; i++) {
        Py_XDECREF(args->op[i]);
    }
    Py_XDECREF(ret);
    return NULL;
}

static PyObject *
py_noise2(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject* dim_vals[2] = {NULL, NULL};
    PyArrayObject *op[3] = {NULL, NULL, NULL};
    npy_uint32 op_flags[3];
    PyArray_Descr *op_dtypes[3];

    NoiseArgs nargs = {
        2,
        3,
        dim_vals,
        op,
        op_flags,
        op_dtypes,
        noise2_scalar,
        dispatch_noise2_args,
        1,
        0.5f,
        2.0f,
        FLT_MAX,
        FLT_MAX,
        0.0f
    };

    static char *kwlist[] = {"x", "y", "octaves", "persistence", "lacunarity",
                             "repeatx", "repeaty", "base", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|ifffff:snoise2",
                                     kwlist,
                                     &dim_vals[0], &dim_vals[1],
                                     &nargs.octaves, &nargs.persistence,
                                     &nargs.lacunarity, &nargs.repeatx,
                                     &nargs.repeaty, &nargs.z))
        return NULL;

    return py_noise_common(&nargs);
}

static PyObject *
py_noise3(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject* dim_vals[3] = {NULL, NULL, NULL};
    PyArrayObject *op[4] = {NULL, NULL, NULL, NULL};
    npy_uint32 op_flags[4];
    PyArray_Descr *op_dtypes[4];

    NoiseArgs nargs = {
        3,
        4,
        dim_vals,
        op,
        op_flags,
        op_dtypes,
        noise3_scalar,
        dispatch_noise3_args,
        1,
        0.5f,
        2.0f
    };

    static char *kwlist[] = {"x", "y", "z", "octaves", "persistence",
                             "lacunarity", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|iff:snoise3",
                                     kwlist,
                                     &dim_vals[0], &dim_vals[1], &dim_vals[2],
                                     &nargs.octaves, &nargs.persistence,
                                     &nargs.lacunarity))
        return NULL;

    return py_noise_common(&nargs);
}

static PyObject *
py_noise4(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyObject* dim_vals[4] = {NULL, NULL, NULL, NULL};
    PyArrayObject *op[5] = {NULL, NULL, NULL, NULL, NULL};
    npy_uint32 op_flags[5];
    PyArray_Descr *op_dtypes[5];

    NoiseArgs nargs = {
        4,
        5,
        dim_vals,
        op,
        op_flags,
        op_dtypes,
        noise4_scalar,
        dispatch_noise4_args,
        1,
        0.5f,
        2.0f
    };

    static char *kwlist[] = {"x", "y", "z", "w", "octaves", "persistence",
                             "lacunarity", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|iff:snoise4",
                                     kwlist,
                                     &dim_vals[0], &dim_vals[1], &dim_vals[2],
                                     &dim_vals[3],
                                     &nargs.octaves, &nargs.persistence,
                                     &nargs.lacunarity))
        return NULL;

    return py_noise_common(&nargs);
}

#define SIMPLEX_COMMON_DOCS \
    "octaves -- specifies the number of passes, defaults to 1 (simple noise).\n\n" \
    "persistence -- specifies the amplitude of each successive octave relative\n" \
    "to the one below it. Defaults to 0.5 (each higher octave's amplitude\n" \
    "is halved). Note the amplitude of the first pass is always 1.0.\n\n" \
    "lacunarity -- specifies the frequency of each successive octave relative\n" \
    "to the one below it, similar to persistence. Defaults to 2.0."

static PyMethodDef simplex_functions[] = {
    {"noise2", (PyCFunction)py_noise2, METH_VARARGS | METH_KEYWORDS,
        "noise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=None, repeaty=None, base=0.0) "
        "return simplex noise value for specified 2D coordinate.\n\n"
        "repeatx, repeaty -- specifies the interval along each axis when \n"
        "the noise values repeat. This can be used as the tile size for creating \n"
        "tileable textures\n\n"
        SIMPLEX_COMMON_DOCS "\n\n"
        "base -- specifies a fixed offset for the noise coordinates. Useful for\n"
        "generating different noise textures with the same repeat interval"},
    {"noise3", (PyCFunction)py_noise3, METH_VARARGS | METH_KEYWORDS,
        "noise3(x, y, z, octaves=1, persistence=0.5, lacunarity=2.0) return simplex noise value for "
        "specified 3D coordinate\n\n"
        SIMPLEX_COMMON_DOCS
    },
    {"noise4", (PyCFunction)py_noise4, METH_VARARGS | METH_KEYWORDS,
        "noise4(x, y, z, w, octaves=1, persistence=0.5, lacunarity=2.0) return simplex noise value for "
        "specified 4D coordinate\n\n"
        SIMPLEX_COMMON_DOCS
    },
    {NULL}
};

#undef SIMPLEX_COMMON_DOCS

PyDoc_STRVAR(module_doc, "Native-code simplex noise functions");

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_simplex",
    module_doc,
    -1,                 /* m_size */
    simplex_functions,  /* m_methods */
    NULL,               /* m_reload (unused) */
    NULL,               /* m_traverse */
    NULL,               /* m_clear */
    NULL                /* m_free */
};

PyMODINIT_FUNC
PyInit__simplex(void)
{
    PyObject* ret = PyModule_Create(&moduledef);
    if (! ret) {
        return NULL;
    }

    import_array();
    return ret;
}

#else

PyMODINIT_FUNC
init_simplex(void)
{
    Py_InitModule3("_simplex", simplex_functions, module_doc);
    import_array();
}

#endif
