/*
Implement parallel vectorize workqueue on top of Intel TBB.
*/

#define TBB_PREVIEW_WAITING_FOR_WORKERS 1
/* tbb.h redefines these */
#include "../../_pymodule.h"
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <tbb/version.h>
#include <tbb/tbb.h>
#include <string.h>
#include <stdio.h>
#include <thread>
#include "workqueue.h"

#include "parallel_runtime.h"

/* TBB 2021.6 is the minimum version */
#if (TBB_INTERFACE_VERSION < 12060)
#error "TBB version is incompatible, 2021.6 or greater required, i.e. TBB_INTERFACE_VERSION >= 12060"
#endif

#define _DEBUG 0
#define _TRACE_SPLIT 0

#ifndef PYCC_PARALLEL
static tbb::task_group *tg = NULL;
#endif

#ifdef _MSC_VER
#define THREAD_LOCAL(ty) __declspec(thread) ty
#else
/* Non-standard C99 extension that's understood by gcc and clang */
#define THREAD_LOCAL(ty) __thread ty
#endif

// This is the number of threads that is default, it is set on initialisation of
// the threading backend via the launch_threads() call
static int _INIT_NUM_THREADS = -1;

// This is the per-thread thread mask, each thread can carry its own mask.
static THREAD_LOCAL(int) _TLS_num_threads = 0;


static void
set_num_threads(int count)
{
    _TLS_num_threads = count;
}

static int
get_num_threads(void)
{
    if (_TLS_num_threads == 0)
    {
        // This is a thread that did not call launch_threads() but is still a
        // "main" thread, probably from e.g. threading.Thread() use, it still
        // has a TLS slot which is 0 from the lack of launch_threads() call
        _TLS_num_threads = _INIT_NUM_THREADS;
    }
    return _TLS_num_threads;
}

static int
get_thread_id(void)
{
    int tid = tbb::this_task_arena::current_thread_index();
    // This function may be called from pure python or from a sequential JIT
    // region, in either case, the task_arena may not be initialised. This
    // condition is intercepted and a thread ID of 0 is returned.
    if (tid == tbb::task_arena::not_initialized)
    {
        return 0;
    } else {
        return tid;
    }
}

#ifndef PYCC_PARALLEL
static void
add_task(void *fn, void *args, void *dims, void *steps, void *data)
{
    tg->run([=]
    {
        auto func = reinterpret_cast<void (*)(void *args, void *dims, void *steps, void *data)>(fn);
        func(args, dims, steps, data);
    });
}
#endif

static void
parallel_for(void *fn, char **args, size_t *dimensions, size_t *steps, void *data,
             size_t inner_ndim, size_t array_count, int num_threads)
{
    static bool printed = false;
    if(!printed && _DEBUG)
    {
        puts("Using parallel_for");
        printed = true;
    }

    //     args = <ir.Argument '.1' of type i8**>,
    //     dimensions = <ir.Argument '.2' of type i64*>
    //     steps = <ir.Argument '.3' of type i64*>
    //     data = <ir.Argument '.4' of type i8*>

    const size_t arg_len = (inner_ndim + 1);

    if(_DEBUG && _TRACE_SPLIT)
    {
        printf("inner_ndim: %lu\n",inner_ndim);
        printf("arg_len: %lu\n", arg_len);
        printf("total: %lu\n", dimensions[0]);
        printf("dimensions: ");
        for(size_t j = 0; j < arg_len; j++)
            printf("%lu, ", ((size_t *)dimensions)[j]);
        printf("\nsteps: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%lu, ", steps[j]);
        printf("\n*args: ");
        for(size_t j = 0; j < array_count; j++)
            printf("%p, ", (void *)args[j]);
        printf("\n");
    }

    tbb::task_arena limited(num_threads);

    limited.execute([&]{
        using range_t = tbb::blocked_range<size_t>;
        tbb::parallel_for(range_t(0, dimensions[0]), [=](const range_t &range)
        {
            size_t * count_space = (size_t *)alloca(sizeof(size_t) * arg_len);
            char ** array_arg_space = (char**)alloca(sizeof(char*) * array_count);
            memcpy(count_space, dimensions, arg_len * sizeof(size_t));
            count_space[0] = range.size();

            if(_DEBUG && _TRACE_SPLIT > 1)
            {
                printf("THREAD %p:", count_space);
                printf("count_space: ");
                for(size_t j = 0; j < arg_len; j++)
                    printf("%lu, ", count_space[j]);
                printf("\n");
            }
            for(size_t j = 0; j < array_count; j++)
            {
                char * base = args[j];
                size_t step = steps[j];
                ptrdiff_t offset = step * range.begin();
                array_arg_space[j] = base + offset;

                if(_DEBUG && _TRACE_SPLIT > 2)
                {
                    printf("Index %ld\n", j);
                    printf("-->Got base %p\n", (void *)base);
                    printf("-->Got step %lu\n", step);
                    printf("-->Got offset %ld\n", offset);
                    printf("-->Got addr %p\n", (void *)array_arg_space[j]);
                }
            }

            if(_DEBUG && _TRACE_SPLIT > 2)
            {
                printf("array_arg_space: ");
                for(size_t j = 0; j < array_count; j++)
                    printf("%p, ", (void *)array_arg_space[j]);
                printf("\n");
            }
            auto func = reinterpret_cast<void (*)(char **args, size_t *dims, size_t *steps, void *data)>(fn);
            int saved = _TLS_num_threads;
            set_num_threads(num_threads);
            func(array_arg_space, count_space, steps, data);
            set_num_threads(saved);
        });
    });
}

#ifndef _WIN32
struct ForkState {
    tbb::task_scheduler_handle handle;
    std::thread::id owner;
    bool unsafe_child = false;
};

static ForkState local_fork_state;
static ForkState *fork_state;
static thread_local bool fork_is_safe;

static void prepare_fork(void)
{
    fork_is_safe = false;
    if (local_fork_state.unsafe_child) {
        return;
    }
    if (std::this_thread::get_id() != local_fork_state.owner) {
        fprintf(stderr, "Numba: Attempted to fork from a non-main thread, "
                        "the TBB library may be in an invalid state in the "
                        "child process.\n");
        return;
    }

    try {
        if (!local_fork_state.handle) {
            local_fork_state.handle = tbb::attach();
        }
        fork_is_safe = tbb::finalize(local_fork_state.handle, std::nothrow);
    } catch (...) {
        fork_is_safe = false;
    }
    if (!fork_is_safe) {
        local_fork_state.handle.release();
        fprintf(stderr, "Numba: Unable to shut down TBB before fork().\n");
    }
}

static void resume_child(void)
{
    local_fork_state.unsafe_child = !fork_is_safe;
}

static int init_fork_state(void)
{
    // JIT and all AOT extensions using the same TBB runtime share one handle.
    const char *key = "_numba_tbb_fork_state_v1";
    PyObject *state = PySys_GetObject(key);
    if (state != NULL) {
        fork_state = static_cast<ForkState *>(PyCapsule_GetPointer(state, key));
        return fork_state == nullptr ? -1 : 0;
    }

    local_fork_state.handle = tbb::attach();
    local_fork_state.owner = std::this_thread::get_id();
    state = PyCapsule_New(&local_fork_state, key, NULL);
    if (state == NULL) {
        return -1;
    }
    int result = PySys_SetObject(key, state);
    Py_DECREF(state);
    if (result < 0) {
        return -1;
    }
    if (pthread_atfork(prepare_fork, NULL, resume_child) != 0) {
        PySys_SetObject(key, NULL);
        PyErr_SetString(PyExc_RuntimeError, "Could not install TBB fork handlers");
        return -1;
    }
    fork_state = &local_fork_state;
    return 0;
}
#endif

static const char *fork_error(void)
{
#ifndef _WIN32
    if (fork_state != nullptr && fork_state->unsafe_child) {
        return "Numba cannot use TBB in this forked child; use spawn";
    }
#endif
    return nullptr;
}

int numba_parallel_init(int count)
{
    if (_INIT_NUM_THREADS > 0) {
        return 0;
    }
    try {
#ifndef _WIN32
        if (init_fork_state() < 0) {
            return -1;
        }
#endif
#ifndef PYCC_PARALLEL
        tg = new tbb::task_group;
#endif
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Could not initialize Numba TBB");
        return -1;
    }
    _INIT_NUM_THREADS = count;
    return 0;
}

NUMBA_DEFINE_PARALLEL_BACKEND(tbb::info::default_concurrency(), fork_error())

#ifndef PYCC_PARALLEL
static void unload_tbb(void *)
{
    if (tg != NULL) {
        tg->wait();
        delete tg;
        tg = NULL;
    }
}

static void launch_threads(int count)
{
    PyGILState_STATE gil = PyGILState_Ensure();
    if (numba_parallel_init(count) < 0) {
        PyErr_WriteUnraisable(NULL);
    }
    PyGILState_Release(gil);
}

static void synchronize(void)
{
    tg->wait();
}

static void ready(void)
{
}

MOD_INIT(tbbpool)
{
    PyObject *m;
    MOD_DEF(m, "tbbpool", "No docs", NULL)
    if (m == NULL)
        return MOD_ERROR_VAL;
    PyModuleDef *md = PyModule_GetDef(m);
    if (md)
    {
        md->m_free = unload_tbb;
    }

    NUMBA_PARALLEL_SYMBOLS(m);
    SetAttrStringFromVoidPointer(m, launch_threads);
    SetAttrStringFromVoidPointer(m, synchronize);
    SetAttrStringFromVoidPointer(m, ready);
    SetAttrStringFromVoidPointer(m, add_task);
    SetAttrStringFromVoidPointer(m, parallel_for);
    SetAttrStringFromVoidPointer(m, do_scheduling_signed);
    SetAttrStringFromVoidPointer(m, do_scheduling_unsigned);
    SetAttrStringFromVoidPointer(m, set_num_threads);
    SetAttrStringFromVoidPointer(m, get_num_threads);
    SetAttrStringFromVoidPointer(m, get_thread_id);
    SetAttrStringFromVoidPointer(m, set_parallel_chunksize);
    SetAttrStringFromVoidPointer(m, get_parallel_chunksize);
    SetAttrStringFromVoidPointer(m, get_sched_size);
    SetAttrStringFromVoidPointer(m, allocate_sched);
    SetAttrStringFromVoidPointer(m, deallocate_sched);

    return MOD_SUCCESS_VAL(m);
}
#endif
