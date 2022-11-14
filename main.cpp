#include "Thread_Pool.hpp"
#include "Test.hpp"
#define NUM_THREADS 4

template <typename T>
void argmax_mat(const T* tensor_ptr, T* const mat_ptr, const unsigned int num_filters, const unsigned int size)
{
    for(unsigned int i=0; i<size; i++)
    {
        mat_ptr[i] = argmax(tensor_ptr, num_filters);
        tensor_ptr += num_filters;
    }
}

int main()
{
    const unsigned int num_rows = 224;
    const unsigned int num_columns = 224;
    const unsigned int num_filters = 21;
    unsigned const int seed = 0;

    const unsigned int size = num_rows * num_columns * num_filters;
    const unsigned int mat_size = num_rows * num_columns;

    std::vector<int8_t> tensor(size);
    std::vector<int8_t> mat(mat_size);
    std::vector<int8_t> sample_mat(mat_size);

    srand(seed);
    for(auto& i : tensor)
    {
        i = rand()%256 - 128;
    }

    obj_detect::Thread_Pool thread_pool(NUM_THREADS);

    const unsigned int work_per_thread = mat_size/NUM_THREADS;
    const unsigned int work_left = mat_size%NUM_THREADS;
    unsigned int i = 0;
    for(i=0; i<NUM_THREADS; i++)
    {
        thread_pool.assign([&, i](){
            argmax_mat(
                tensor.data() + num_filters*work_per_thread*i, 
                mat.data() + work_per_thread*i, 
                num_filters, 
                (i < work_left ? work_per_thread + 1 : work_per_thread));
        });
    }

    thread_pool.wait_until(NUM_THREADS);

    argmax_mat(tensor.data(), sample_mat.data(), num_filters, mat_size);
    comp_vec(mat, sample_mat);

    //print_tensor(tensor.data(), num_rows, num_columns, num_filters, true);
    //print_tensor(mat.data(), num_rows, num_columns, 1, true);

    return 0;
}