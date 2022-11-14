#pragma once

#include <cstring>

template<typename T>
inline unsigned int argmax(const T* const arr_ptr, unsigned const int size)
{
    const T* max_val_ptr = arr_ptr;
    for(unsigned int i = 1; i<size; i++)
    {
        max_val_ptr = arr_ptr[i] > *max_val_ptr ? (arr_ptr + i) : max_val_ptr;
    }
    return (max_val_ptr - arr_ptr);
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