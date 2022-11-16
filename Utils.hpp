#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <regex>
#include <fstream>
#include <sstream>

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

void fill_vec(std::vector<int8_t>& vec)
{
    for(auto& item : vec)
    {
        item = rand()%256 - 128;
    }
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
