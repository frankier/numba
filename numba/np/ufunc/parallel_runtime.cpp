#include "../../_pymodule.h"
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <mutex>
#include "parallel_runtime.h"

int numba_parallel_init_from_env(void)
{
    const char *value = std::getenv("NUMBA_NUM_THREADS");
    if (value == NULL) {
        return numba_parallel_init(numba_parallel_backend_default_threads());
    }

    char *end;
    errno = 0;
    long count = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE ||
        count < 1 || count > INT_MAX) {
        PyErr_SetString(PyExc_ValueError,
                        "NUMBA_NUM_THREADS must be a positive int");
        return -1;
    }
    return numba_parallel_init(static_cast<int>(count));
}

using kernel_fn = void (*)(char **, size_t *, size_t *, void *);

struct ParallelDispatch {
    explicit ParallelDispatch(kernel_fn fn) : kernel(fn), error{} {}
    kernel_fn kernel;
    PyObject *error[3];
    std::mutex error_mutex;
};

void numba_parallel_save_error(void *data)
{
    auto *dispatch = static_cast<ParallelDispatch *>(data);
    std::lock_guard<std::mutex> lock(dispatch->error_mutex);
    auto &error = dispatch->error;
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Numba parallel worker failed");
    }
    if (error[0] == NULL) {
        PyErr_Fetch(error, error + 1, error + 2);
    } else {
        PyErr_Clear();
    }
}

static void run_kernel(char **args, size_t *dims, size_t *steps, void *data)
{
    auto *dispatch = static_cast<ParallelDispatch *>(data);
    uintp saved_chunksize = numba_parallel_set_chunksize(0);
    dispatch->kernel(args, dims, steps, data);
    numba_parallel_set_chunksize(saved_chunksize);
}

int numba_parallel_dispatch(void *fn, char **args, size_t *dims,
                            size_t *steps, size_t inner_ndim,
                            size_t array_count, int count)
{
    ParallelDispatch dispatch(reinterpret_cast<kernel_fn>(fn));
    PyGILState_STATE gil = PyGILState_Ensure();
    PyThreadState *thread_state = PyEval_SaveThread();
    const char *dispatch_error;
    try {
        dispatch_error = numba_parallel_backend_for(
            reinterpret_cast<void *>(run_kernel), args, dims, steps,
            &dispatch, inner_ndim, array_count, count);
    } catch (...) {
        dispatch_error = "Numba parallel dispatch failed";
    }
    PyEval_RestoreThread(thread_state);
    bool worker_failed = dispatch.error[0] != NULL;
    if (worker_failed) {
        PyErr_Restore(dispatch.error[0], dispatch.error[1], dispatch.error[2]);
    } else if (dispatch_error != NULL) {
        PyErr_SetString(PyExc_RuntimeError, dispatch_error);
    }
    PyGILState_Release(gil);
    return worker_failed || dispatch_error != NULL;
}
