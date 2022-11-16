#pragma once

#include <cstring>
#include "Thread_Pool.hpp"

template<typename T>
inline unsigned int argmax(const T* const arr_ptr, unsigned const int size)
{
    const T* max_val_ptr = arr_ptr;
    for(unsigned int i = 1; i<size; i++)
    {
        max_val_ptr = arr_ptr[i] > *max_val_ptr ? (arr_ptr + i) : max_val_ptr;
    }
    return (unsigned int)(max_val_ptr - arr_ptr);
}

template <typename T>
inline void argmax_tensor(const T* tensor_ptr, T* const mat_ptr, const unsigned int num_filters, const unsigned int mat_size)
{
    for(unsigned int i=0; i<mat_size; i++)
    {
        mat_ptr[i] = (T)argmax(tensor_ptr, num_filters);
        tensor_ptr += num_filters;
    }
}

template <typename T>
void argmax_tensor_mt(
    const T* tensor_ptr, 
    T* const mat_ptr, 
    const unsigned int num_filters, 
    const unsigned int mat_size, 
    obj_detect::Thread_Pool& thread_pool)
{
    const unsigned int num_threads = thread_pool.get_num_threads();
    const unsigned int work_per_thread = mat_size/num_threads;
    const unsigned int work_left = mat_size%num_threads;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;
    for(unsigned int i=0; i<num_threads; i++)
    {
        work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
        thread_pool.assign([&, total_work_count, work_count](){
            argmax_tensor(
                tensor_ptr + num_filters*total_work_count, 
                mat_ptr + total_work_count, 
                num_filters, 
                work_count);
        });
        total_work_count += work_count;
    }
    thread_pool.wait_until(num_threads);
}

template<typename T>
void upsampler(
    const T* const tensor_ptr, 
    T* const scaled_up_tensor_ptr, 
    const unsigned int num_rows, 
    const unsigned int num_columns, 
    const unsigned int num_filters, 
    const unsigned int scale_up_factor)
{
    enum dim{rows=0, columns=1, filters=2};
    const unsigned int shape[] = {num_rows * scale_up_factor, num_columns * scale_up_factor, num_filters}; 

    const unsigned int num_items_per_mat_cell =  shape[dim::filters];
    const unsigned int num_items_per_mat_row = shape[dim::columns] * num_items_per_mat_cell;
    const T* arr_cptr = tensor_ptr;
    T* new_arr_cptr = scaled_up_tensor_ptr;
    for(unsigned int r=0; r<num_rows; r++)
    {
        for(unsigned int c=0; c<num_columns; c++)
        {
            for(unsigned int i=0; i<scale_up_factor; i++)
            {
                memcpy(new_arr_cptr, arr_cptr, sizeof(T) * num_items_per_mat_cell);
                new_arr_cptr += num_items_per_mat_cell;
            }
            arr_cptr += num_items_per_mat_cell;
        }
        for(unsigned int i=0; i<scale_up_factor-1; i++)
        {
            memcpy(new_arr_cptr, new_arr_cptr - num_items_per_mat_row, sizeof(T) * num_items_per_mat_row);
            new_arr_cptr += num_items_per_mat_row;
        }
    }
}
