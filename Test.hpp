#pragma once

#include <iostream>
#include <iomanip>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>

#include "Thread_Pool.hpp"
#include "Utils.hpp"
#include "Timer.hpp"

#define NUM_THREADS 4

static unsigned int cycles = 1000;

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
    const unsigned int cycles,
    unsigned const int seed
)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);

    srand(seed);
    for(unsigned int c=0; c<cycles; c++)
    {
        for(auto& i : tensor)
        {
            i = rand()%256 - 128;
        }

        Timer::Get().start("Argmax -" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        argmax_tensor(tensor.data(), mat.data(), num_filters, mat_size);
        Timer::Get().stop();
    }
}

void argmax_win_benchmark(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles,
    unsigned const int seed
)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int new_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(new_size);

    srand(seed);
    for(unsigned int c=0; c<cycles; c++)
    {
        for(auto& i : tensor)
        {
            i = rand()%256 - 128;
        }

        Timer::Get().start("Argmax win-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        const int8_t* ptr = tensor.data();
        for(auto& i : mat)
        {
            i = std::max_element(ptr, ptr + num_filters) - ptr;
            ptr += num_filters;
        }
        Timer::Get().stop();
    }

    // print_tensor(tensor.data(), num_rows, num_columns, num_filters);
    // print_tensor(mat.data(), num_rows, num_columns, 1);
}

void argmax_mt_benchmark(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int cycles,
    unsigned const int seed
)
{
    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);

    srand(seed);
    obj_detect::Thread_Pool thread_pool(NUM_THREADS);
    const unsigned int work_per_thread = mat_size/NUM_THREADS;
    const unsigned int work_left = mat_size%NUM_THREADS;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;
    
    for(unsigned int c=0; c<cycles; c++)
    {
        for(auto& i : tensor)
        {
            i = rand()%256 - 128;
        }

        Timer::Get().start("Argmax MT-" + std::to_string(num_columns) + "x" + std::to_string(num_rows) + "x" + std::to_string(num_filters));
        total_work_count = 0;
        for(unsigned int i=0; i<NUM_THREADS; i++)
        {
            work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
            thread_pool.assign([&, total_work_count, work_count](){
                argmax_tensor(
                    tensor.data() + num_filters*total_work_count, 
                    mat.data() + total_work_count, 
                    num_filters, 
                    work_count);
            });
            total_work_count += work_count;
        }

        thread_pool.wait_until(NUM_THREADS);
        Timer::Get().stop();
    }
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
    // Important : allocate memory
    std::vector<int> new_vec(new_size);

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
    const unsigned int cycles,
    unsigned const int seed
)
{
    const unsigned int size = num_rows * num_columns * num_filters;

    const unsigned int new_num_rows = num_rows * scale_up_factor;
    const unsigned int new_num_columns = num_columns * scale_up_factor;
    const unsigned int new_num_filters = num_filters;
    const unsigned int new_size = new_num_rows * new_num_columns * new_num_filters;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> new_tensor(new_size);

    srand(seed);
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

void benchmark(unsigned const int seed)
{
    argmax_benchmark(224, 224, 21, cycles, seed);
    argmax_win_benchmark(224, 224, 21, cycles, seed);
    argmax_mt_benchmark(224, 224, 21, cycles, seed);
    argmax_benchmark(28, 28, 21, cycles, seed);
    argmax_win_benchmark(28, 28, 21, cycles, seed);
    argmax_mt_benchmark(28, 28, 21, cycles, seed);

    upsampler_benchmark(28, 28, 21, 8, cycles, seed);
    upsampler_benchmark(28, 28, 1, 8, cycles, seed);
}

std::vector<int8_t> sim_up_scale_argmax(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int scale_up_factor,
    const unsigned int cycles,
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

    srand(seed);

    obj_detect::Thread_Pool thread_pool(NUM_THREADS);
    const unsigned int work_per_thread = scaled_up_mat_size/NUM_THREADS;
    const unsigned int work_left = scaled_up_mat_size%NUM_THREADS;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;

    for(unsigned int c=0; c<cycles;c++)
    {
        for(auto& item : tensor)
        {
            item = rand()%256 - 128;
        }
        
        Timer::Get().start("up scale->argmax");
        upsampler(tensor.data(), scaled_up_tensor.data(), num_rows, num_columns, num_filters, scale_up_factor);
        total_work_count = 0;
        for(unsigned int i=0; i<NUM_THREADS; i++)
        {
            work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
            thread_pool.assign([total_work_count, work_count, &num_filters, &scaled_up_tensor, &scaled_up_mat](){
                argmax_tensor(
                    scaled_up_tensor.data() + num_filters*total_work_count, 
                    scaled_up_mat.data() + total_work_count, 
                    num_filters, 
                    work_count);
            });
            total_work_count += work_count;
        }

        thread_pool.wait_until(NUM_THREADS);
        Timer::Get().stop();
    }

    return scaled_up_mat;
}

std::vector<int8_t> sim_argmax_up_scale(
    const unsigned int num_rows,
    const unsigned int num_columns,
    const unsigned int num_filters,
    const unsigned int scale_up_factor,
    const unsigned int cycles,
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

    srand(seed);

    obj_detect::Thread_Pool thread_pool(NUM_THREADS);
    const unsigned int work_per_thread = mat_size/NUM_THREADS;
    const unsigned int work_left = mat_size%NUM_THREADS;
    unsigned int total_work_count = 0;
    unsigned int work_count = 0;
    
    for(unsigned int c=0; c<cycles;c++)
    {
        for(auto& item : tensor)
        {
            item = rand()%256 - 128;
        }
    
        Timer::Get().start("argmax->up scale");
        total_work_count = 0;
        for(unsigned int i=0; i<NUM_THREADS; i++)
        {
            work_count = (i < work_left ? work_per_thread + 1 : work_per_thread);
            thread_pool.assign([total_work_count, work_count, &num_filters, &tensor, &mat](){
                argmax_tensor(
                    tensor.data() + num_filters*total_work_count, 
                    mat.data() + total_work_count, 
                    num_filters, 
                    work_count);
            });
            total_work_count += work_count;
        }

        thread_pool.wait_until(NUM_THREADS);

        upsampler(mat.data(), scaled_up_mat.data(), num_rows, num_columns, 1, scale_up_factor);
        Timer::Get().stop();
    }

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
            std::cerr<<"value mismatch : "<< (int)vec_1[i]<< " != " << (int)vec_2[i] <<std::endl;
            return;
        }
    }
}

void test()
{
    //argmax_example();
    //upsampler_example();

    unsigned const int seed = time(NULL);
    std::vector<int8_t> sim_1_out = sim_up_scale_argmax(28, 28, 21, 8, cycles, seed);
    std::vector<int8_t> sim_2_out = sim_argmax_up_scale(28, 28, 21, 8, cycles, seed);
    comp_vec(sim_1_out, sim_2_out);

    benchmark(seed);

    Timer::Get().print_duration();
    Timer::Get().reset();
}

std::vector<int> array_dimensions(std::vector<std::string> line_array) 
{
    std::string s = line_array[0];
    std::string delimiter_1 = "(";
    std::string delimiter_2 = ")";
    int str_start = s.find(delimiter_1) + 1;
    int str_end = s.find(delimiter_2) - str_start;
    std::string token = s.substr(str_start, str_end);
    token = std::regex_replace(token, std::regex(","), "");

    std::vector<int> arr;
    std::stringstream sstream(token);
    int temp;
    while (sstream >> temp)
        arr.push_back(temp);

    return arr;
}

void vector_populator(
    const std::string name,
    std::vector<int8_t>& vec,
    unsigned int& width,
    unsigned int& height,
    unsigned int& channel) 
{

    std::vector<std::string> arr;
    std::vector<int> dim_array;

    std::ifstream file(name);
    if (!file.is_open())
    {
        std::cerr << "File not opened..\n";
        exit(1);
    }

    std::string str;
    while (getline(file, str))
    {
        arr.push_back(str);
    }

    dim_array = array_dimensions(arr);

    width = dim_array[2];
    height = dim_array[1];
    channel = dim_array[3];

    int count = 1;
    for (unsigned int i = 0; i < height; i++) {
        for (unsigned int j = 0; j < width; j++) {
            for (unsigned int k = 0; k < channel; k++) {
                vec.push_back(std::stof(arr[count]));
                count++;
            }
        }
    }
}

void sim_model_outputs()
{
    std::vector<int8_t> input_tensor;
    unsigned int input_width = 0;
    unsigned int input_height = 0;
    unsigned int input_channel = 0;
    std::vector<int8_t> output_tensor;
    unsigned int output_width = 0;
    unsigned int output_height = 0;
    unsigned int output_channel = 0;

    vector_populator("..\\..\\fcn224_data\\o_62.txt", input_tensor, input_width, input_height, input_channel);
    vector_populator("..\\..\\fcn224_data\\o_64.txt", output_tensor, output_width, output_height, output_channel);

    std::vector<int8_t> mat(input_width*input_height);
    const int8_t* ptr = input_tensor.data();
    for(auto& i : mat)
    {
        i = argmax(ptr, input_channel);
        ptr += input_channel;
    }

    std::vector<int8_t> output_tensor_algo(224*224);
    upsampler(mat.data(), output_tensor_algo.data(), input_width, input_height, 1, 8);

    comp_vec(output_tensor, output_tensor_algo);
}
