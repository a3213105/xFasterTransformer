// Copyright (c) 2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#define THREAD_TYPE_TBB      0
#define THREAD_TYPE_OMP      1
#define THREAD_TYPE_SEQ      2

#ifndef THREAD_TYPE
#define THREAD_TYPE THREAD_TYPE_TBB
#endif

#if THREAD_TYPE == THREAD_TYPE_TBB
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef TBB_PREVIEW_LOCAL_OBSERVER
#        define TBB_PREVIEW_LOCAL_OBSERVER 1
#    endif
#    ifndef TBB_PREVIEW_WAITING_FOR_WORKERS
#        define TBB_PREVIEW_WAITING_FOR_WORKERS 1
#    endif
#    ifndef TBB_PREVIEW_NUMA_SUPPORT
#        define TBB_PREVIEW_NUMA_SUPPORT 1
#    endif
#    ifndef TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION
#        define TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION 1
#    endif
#    ifndef TBB_PREVIEW_BLOCKED_RANGE_ND
#        define TBB_PREVIEW_BLOCKED_RANGE_ND 1
#    endif
#    include "tbb/blocked_range.h"
#    include "tbb/blocked_range2d.h"
#    include "tbb/blocked_range3d.h"
#    include "tbb/blocked_nd_range.h"
#    include "tbb/parallel_for.h"
#    include "tbb/parallel_reduce.h"
#    include "tbb/parallel_sort.h"
#    include "tbb/info.h"
#    include "tbb/task_arena.h"
#    include "tbb/spin_mutex.h"
#    include "tbb/combinable.h"

inline int parallel_get_max_threads() {
    return tbb::this_task_arena::max_concurrency();
}
inline int parallel_get_num_threads() {
    return tbb::this_task_arena::max_concurrency();
}
inline int parallel_get_thread_num() {
    return tbb::this_task_arena::current_thread_index();
}
inline void parallel_set_num_threads(int num_threads) {
    return ;
}

#elif THREAD_TYPE == THREAD_TYPE_OMP
#    include <omp.h>

#    include <algorithm>
#    include <cstdlib>
#    include <string>

inline int parallel_get_max_threads() {
    // auto origin_dyn_val = omp_get_dynamic();
    // if (origin_dyn_val != 0) {
    //     omp_set_dynamic(0);
    // }
    return omp_get_max_threads();
}
inline int parallel_get_num_threads() {
    return omp_get_num_threads();
}
inline int parallel_get_thread_num() {
    return omp_get_thread_num();
}
inline void parallel_set_num_threads(int num_threads) {
    omp_set_num_threads(num_threads);
}
#elif THREAD_TYPE == THREAD_TYPE_SEQ
#    include <algorithm>
inline int parallel_get_env_threads() {
    return 1;
}
inline int parallel_get_max_threads() {
    return 1;
}
inline int parallel_get_num_threads() {
    return 1;
}
inline void parallel_set_num_threads(int num_threads) {
    return ;
}
#endif
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>

namespace xft {

template <typename Lambda>
void parallel_for(int tasks, const Lambda &fn) {
#if THREAD_TYPE == THREAD_TYPE_TBB
tbb::parallel_for(tbb::blocked_range<int>(0, tasks),
[&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for
#   endif
    for (int i = 0; i < tasks; i++) {
#endif
        fn(i);
    }
#if THREAD_TYPE == THREAD_TYPE_TBB
    });
#endif
}

template <typename Lambda>
void parallel_for_dschedule(int tasks, const Lambda &fn) {
#if THREAD_TYPE == THREAD_TYPE_TBB
tbb::parallel_for(tbb::blocked_range<int>(0, tasks),
[&](const tbb::blocked_range<int>& range) {
    for (int i = range.begin(); i < range.end(); ++i) {
#else
#   if THREAD_TYPE == THREAD_TYPE_OMP
#   pragma omp parallel for schedule(dynamic)
#   endif
    for (int i = 0; i < tasks; i++) {
#endif
        fn(i);
    }
#if THREAD_TYPE == THREAD_TYPE_TBB
    });
#endif
}

class ThreadPool {
public:
    static ThreadPool &getInstance() {
        static ThreadPool instance;
        return instance;
    }

    template <typename F, typename... Args>
    void addTask(F &&f, Args &&...args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

private:
    ThreadPool() : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) { return; }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    static constexpr size_t numThreads = 1;
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

} // namespace xft