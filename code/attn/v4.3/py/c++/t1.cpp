#include <iostream>


#include <algorithm>
#include <limits>
#include <sstream>
#include <vector>


#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xnpy.hpp>


#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/generators/xbuilder.hpp>
#include <xtensor/generators/xrandom.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xdynamic_view.hpp>
#include <xtensor/views/xview.hpp>



#include <stdfloat> // from c++23
using bfloat16_t = std::bfloat16_t;

typedef bfloat16_t bf16;


// not work on mac arm64
//#include <cnpy.h>


/*

  //g++ -std=c++20 -I./xtensor/include -I./xtl/include -I./cnpy ./cnpy/cnpy.cpp t1.cpp

  g++ -std=c++20 -I./xtensor/include -I./xtl/include  t1.cpp

g++ -std=c++23 -I./xtensor/include -I./xtl/include  t1.cpp

*/



int main(void)
{

    auto arr2 = xt::load_npy<float>("q_np_f32.npy");

    //std::cout << "Loaded:\n" << arr2 << std::endl;
    for (auto v : arr2) {
        bf16 b = (bf16)v;
        //printf("0x%x \n", b);
        //std::cout<<b<<std::endl;
    }



    return 0;
}
