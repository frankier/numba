#ifndef NUMBA_PARALLEL_RUNTIME_H
#define NUMBA_PARALLEL_RUNTIME_H

#include <stddef.h>
#include "../../_numba_common.h"
#include "gufunc_scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

VISIBILITY_HIDDEN int numba_parallel_init(int count);
VISIBILITY_HIDDEN int numba_parallel_init_from_env(void);
VISIBILITY_HIDDEN int numba_parallel_get_num_threads(void);
VISIBILITY_HIDDEN int numba_parallel_get_max_threads(void);
VISIBILITY_HIDDEN int numba_parallel_get_thread_id(void);
VISIBILITY_HIDDEN void numba_parallel_set_num_threads(int count);
VISIBILITY_HIDDEN uintp numba_parallel_get_chunksize(void);
VISIBILITY_HIDDEN uintp numba_parallel_set_chunksize(uintp count);
VISIBILITY_HIDDEN intp *numba_parallel_schedule_signed(
    uintp ndim, intp *starts, intp *ends, uintp count, intp debug,
    uintp *divisions);
VISIBILITY_HIDDEN intp *numba_parallel_schedule_unsigned(
    uintp ndim, intp *starts, intp *ends, uintp count, intp debug,
    uintp *divisions);
VISIBILITY_HIDDEN void numba_parallel_free_schedule(intp *schedule);
VISIBILITY_HIDDEN int numba_parallel_dispatch(void *fn, char **args, size_t *dims,
                            size_t *steps, size_t inner_ndim,
                            size_t array_count, int count);
VISIBILITY_HIDDEN void numba_parallel_save_error(void *error);

VISIBILITY_HIDDEN const char *numba_parallel_backend_for(void *fn, char **args, size_t *dims,
                                       size_t *steps, void *data,
                                       size_t inner_ndim, size_t array_count,
                                       int count);
VISIBILITY_HIDDEN int numba_parallel_backend_default_threads(void);

#ifdef __cplusplus
}
#endif

#define NUMBA_DEFINE_PARALLEL_BACKEND(default_threads, check_dispatch, run)    \
    int numba_parallel_get_num_threads(void) { return get_num_threads(); }    \
    int numba_parallel_get_max_threads(void) { return _INIT_NUM_THREADS; }    \
    int numba_parallel_get_thread_id(void) { return get_thread_id(); }        \
    void numba_parallel_set_num_threads(int n) { set_num_threads(n); }        \
    int numba_parallel_backend_default_threads(void) { return default_threads; } \
    const char *numba_parallel_backend_for(                                   \
        void *fn, char **args, size_t *dims, size_t *steps, void *data,        \
        size_t inner_ndim, size_t array_count, int count)                     \
    {                                                                        \
        const char *error = check_dispatch;                                  \
        if (error == NULL) {                                                 \
            run(fn, args, dims, steps, data, inner_ndim, array_count, count);  \
        }                                                                    \
        return error;                                                        \
    }

#define NUMBA_PARALLEL_SYMBOLS(m)                                             \
    SetAttrStringFromVoidPointer(m, numba_parallel_init);                    \
    SetAttrStringFromVoidPointer(m, numba_parallel_get_num_threads);         \
    SetAttrStringFromVoidPointer(m, numba_parallel_get_max_threads);         \
    SetAttrStringFromVoidPointer(m, numba_parallel_get_thread_id);           \
    SetAttrStringFromVoidPointer(m, numba_parallel_set_num_threads);         \
    SetAttrStringFromVoidPointer(m, numba_parallel_get_chunksize);           \
    SetAttrStringFromVoidPointer(m, numba_parallel_set_chunksize);           \
    SetAttrStringFromVoidPointer(m, numba_parallel_schedule_signed);         \
    SetAttrStringFromVoidPointer(m, numba_parallel_schedule_unsigned);       \
    SetAttrStringFromVoidPointer(m, numba_parallel_free_schedule);           \
    SetAttrStringFromVoidPointer(m, numba_parallel_dispatch);                \
    SetAttrStringFromVoidPointer(m, numba_parallel_save_error)

#endif
