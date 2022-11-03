#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <limits>

#include "Timer.hpp"

template<typename T>
void print_tensor(
    const T* const tensor_ptr,
    const unsigned int num_rows, 
    const unsigned int num_columns, 
    const unsigned int num_filters,
    bool cast_to_int_for_print = false
)
{
    int i=0;
    for(unsigned int r=0; r<num_rows; r++)
    {
        for(unsigned int c=0; c<num_columns; c++)
        {
            std::cout<<"( ";
            for(unsigned int f=0; f<num_filters; f++)
            {
                if(!cast_to_int_for_print) std::cout<<std::setw(4)<<tensor_ptr[i]<<" ";
                else std::cout<<std::setw(4)<<(int)tensor_ptr[i]<<" ";
                i++;
            }
            std::cout<<") ";
        }
        std::cout<<std::endl;
    }
}

template<typename T>
unsigned int argmax(const T* const arr_ptr, unsigned const int size)
{
    unsigned int max_val_index = 0;
    T max_val = std::numeric_limits<T>().min();
    for(unsigned int i = 0; i<size; i++)
    {
        if(arr_ptr[i] > max_val)
        {
            max_val = arr_ptr[i];
            max_val_index = i;
        }
    }
    return max_val_index;
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

void argmax_example()
{
    unsigned int max_i = 0;

    int8_t arr_a[] = {4, 1, 7, 5, 3, 8, 5, 1};
    const int arr_a_size = sizeof(arr_a)/sizeof(int8_t);
    max_i = argmax(arr_a, arr_a_size);
    std::cout<<"Index : "<< max_i <<" - Value : "<<(int)arr_a[max_i]<<std::endl;

    std::array<float, 8> arr_b = {-4.0f, 1.32f, 7.12f, 5.1f, 3.0f, -8.54f, 5.0f, 1.1f};
    max_i = argmax(arr_b.data() , arr_b.size());
    std::cout<<"Index : "<< max_i <<" - Value : "<<arr_b[max_i]<<std::endl;
}

void argmax_benchmark(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles = 1000
)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int new_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(new_size);

    srand(time(NULL));
    for(unsigned int c=0; c<cycles; c++)
    {
        for(auto& i : tensor)
        {
            i = rand()%256 - 128;
        }

        Timer::Get().start("Argmax-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        const int8_t* ptr = tensor.data();
        for(auto& i : mat)
        {
            i = ptr[argmax(ptr, num_filters)];
            ptr += num_filters;
        }
        Timer::Get().stop();
    }

    // print_tensor(tensor.data(), num_rows, num_columns, num_filters);
    // print_tensor(mat.data(), num_rows, num_columns, 1);
}

void upsampler_example()
{
    // r, c, f
    // h, w, c

    // create a dummy array for test
    const int num_rows = 4;
    const int num_columns = 3;
    const int num_filters = 2;
    const int size = num_rows * num_columns * num_filters;
    std::array<int, size> arr;
    // put dummy values
    for(int i=0; i<size; i++)
    {
        arr[i] = i+1;
    }

    // scale-up factor
    const int scale_up_factor = 5;

    // calculate new scaled-up array dimensions
    const int new_num_rows = num_rows * scale_up_factor;
    const int new_num_columns = num_columns * scale_up_factor;
    const int new_num_filters = num_filters;

    // calculate new scaled-up array size
    const int new_size = new_num_rows * new_num_columns * new_num_filters;
    // create a new array for store the scaled-up tensor (allocate memory)
    std::array<int, new_size> new_arr;

    upsampler(arr.data(), new_arr.data(), num_rows, num_columns, num_filters, scale_up_factor);

    std::cout<<"Source tensor :\n";
    print_tensor(arr.data(), num_rows, num_columns, num_filters);
    std::cout<<"Scaled-up tensor :\n";
    print_tensor(new_arr.data(), new_num_rows, new_num_columns, new_num_filters);
    
    // example for vector
    std::vector<int> new_vec;
    // Important : allocate memory
    new_vec.reserve(new_size);

    upsampler(arr.data(), new_vec.data(), num_rows, num_columns, num_filters, scale_up_factor);

    std::cout<<"Source tensor :\n";
    print_tensor(arr.data(), num_rows, num_columns, num_filters);
    std::cout<<"Scaled-up tensor :\n";
    print_tensor(new_vec.data(), new_num_rows, new_num_columns, new_num_filters);
}

void upsampler_benchmark(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int scale_up_factor,
    const unsigned int cycles = 1000
)
{
    const unsigned int size = num_rows * num_columns * num_filters;

    const unsigned int new_num_rows = num_rows * scale_up_factor;
    const unsigned int new_num_columns = num_columns * scale_up_factor;
    const unsigned int new_num_filters = num_filters;
    const unsigned int new_size = new_num_rows * new_num_columns * new_num_filters;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> new_tensor(new_size);

    srand(time(NULL));
    for(unsigned int c=0; c<cycles; c++)
    {
        for(unsigned int i=0; i<size; i++)
        {
            tensor[i] = rand()%256 - 128;
        }
        Timer::Get().start("Upsampler-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters) + "-" +  std::to_string(scale_up_factor));
        upsampler(tensor.data(), new_tensor.data(), num_rows, num_columns, num_filters, scale_up_factor);
        Timer::Get().stop();
    }
}

void benchmark()
{
    argmax_benchmark(224, 224, 21, 1000);
    argmax_benchmark(28, 28, 21, 1000);

    upsampler_benchmark(28, 28, 21, 8, 1000);
    upsampler_benchmark(28, 28, 1, 8, 1000);    
}

std::vector<int8_t> sim_up_scale_argmax(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int scale_up_factor,
    unsigned const int seed)
{
    const unsigned int tensor_size = num_rows * num_columns * num_filters;
    const unsigned int scaled_up_num_rows = num_rows * scale_up_factor;
    const unsigned int scaled_up_num_columns = num_columns * scale_up_factor;
    const unsigned int scaled_up_tensor_size = scaled_up_num_rows * scaled_up_num_columns * num_filters;
    const unsigned int scaled_up_mat_size = scaled_up_num_rows * scaled_up_num_columns;

    std::vector<int8_t> tensor(tensor_size);
    std::vector<int8_t> scaled_up_tensor(scaled_up_tensor_size);
    std::vector<int8_t> scaled_up_mat(scaled_up_mat_size);

    for(int c=0; c<100;c++)
    {
        srand(seed);
        for(auto& item : tensor)
        {
            item = rand()%256 - 128;
        }
        
        Timer::Get().start("up scale->argmax");
        upsampler(tensor.data(), scaled_up_tensor.data(), num_rows, num_columns, num_filters, scale_up_factor);

        const int8_t* ptr = scaled_up_tensor.data();
        for(auto& i : scaled_up_mat)
        {
            i = ptr[argmax(ptr, num_filters)];
            ptr += num_filters;
        }
        Timer::Get().stop();
    }

    //print_tensor(scaled_up_mat.data(), scaled_up_num_rows, scaled_up_num_columns, 1, true);
    //std::cout<<std::endl;

    return scaled_up_mat;
}

std::vector<int8_t> sim_argmax_up_scale(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int scale_up_factor,
    unsigned const int seed)
{
    const unsigned int tensor_size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;
    const unsigned int scaled_up_num_rows = num_rows * scale_up_factor;
    const unsigned int scaled_up_num_columns = num_columns * scale_up_factor;
    const unsigned int scaled_up_mat_size = scaled_up_num_rows * scaled_up_num_columns;

    std::vector<int8_t> tensor(tensor_size);
    std::vector<int8_t> mat(mat_size);
    std::vector<int8_t> scaled_up_mat(scaled_up_mat_size);

    for(int c=0; c<100;c++)
    {
        srand(seed);
        for(auto& item : tensor)
        {
            item = rand()%256 - 128;
        }

        Timer::Get().start("argmax->up scale");
        const int8_t* ptr = tensor.data();
        for(auto& i : mat)
        {
            i = ptr[argmax(ptr, num_filters)];
            ptr += num_filters;
        }
        
        upsampler(mat.data(), scaled_up_mat.data(), num_rows, num_columns, 1, scale_up_factor);
        Timer::Get().stop();
    }

    //print_tensor(scaled_up_mat.data(), scaled_up_num_rows, scaled_up_num_columns, 1, true);
    //std::cout<<std::endl;

    return scaled_up_mat;
}

template <typename T>
void comp_vec(const std::vector<T> vec_1, const std::vector<T> vec_2)
{
    if(vec_1.size() != vec_2.size())
    {
        std::cerr<<"size mismatch\n";
        return;
    }

    for(int i=0; i<vec_1.size(); i++)
    {
        if(vec_1[i] != vec_2[i])
        {
            std::cerr<<"value mismatch\n";
            return;
        }
    }
}

int main()
{
    //argmax_example();
    //upsampler_example();
    //benchmark();

    unsigned const int seed = time(NULL);
    std::vector<int8_t> sim_1_out = sim_up_scale_argmax(28, 28, 21, 8, seed);
    std::vector<int8_t> sim_2_out = sim_argmax_up_scale(28, 28, 21, 8, seed);

    Timer::Get().print_duration();
    Timer::Get().reset();

    comp_vec(sim_1_out, sim_2_out);

    return 0;
}