#include <iostream>
#include <assert.h>
// #include <atmotic>

#include "wm_chess.h"
#include <cfloat>

// void deleter(int *a){
//     std :: cout << "delete " << *a << std :: endl;
//     delete a;
// }

int main(){
    WMChess* wm = new WMChess(7, -1);
    std::vector<int> boad =  std::vector<int>(21,0);
    boad[14] = 1;
    boad[12] = -1;
    boad[15] = -1;
    boad[10] = -1;
    WMChess::move_type move = std::make_pair(10,11);
    wm -> set_board(boad);
    wm->print_origin_board();
    wm->execute_move(move);
    wm->print_origin_board();




    // WMChess::move_type temp = wm->find_row_column_in_map(WMChess::move_type(0,10));
    // std::cout <<  temp.first << "," << temp.second << std::endl;

    // std:: cout << "wm = " << wm << std::endl;
    // wm = nullptr;
    // int c = wm -> WHITE ;
    // std::cout << c << std::endl;

    // std::vector<int> temp;
    // temp.size();

    // std::unique_ptr<int,decltype(deleter)*> temp(new int(100), deleter);
    // std::cout << FLT_EPSILON << " -DBL_MAX = " << -DBL_MAX << std::endl;
    // temp.reset(nullptr);
// int * test = nullptr;
// std::cout << *test << std::endl;
// auto temp = std::make_shared<WMChess>(*wm);

//     wm->set_board(std::vector<int>(21,-1));
// wm->print_board();
// std :: cout << "temp = " << temp.get() << (0 < FLT_EPSILON ?"    0<FLT_EPSILON":"0>=FLT_EPSILON")<<std::endl;
// temp->print_board();

    // WMChess::board_type board_temp = {1,1,0,0,0,0,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0};

    // wm.set_board(board_temp);
    // std::vector<int> result = wm.get_game_status();
    // std::cout << "is end " << result[0] << " winner " << result[1] << std::endl;

    // int a = 1, b = 1000;
    //     std::cout << a << ","<<b<<std::endl;
    // std::swap(a,b);
    // std::cout << a << ","<<b<<std::endl;

    // std::cout << "from " << wm.get_last_move().first << " to " << wm.get_last_move().second << std::endl;

    // wm.get_distance();
    // std::vector<int> temp = {27,42,15,49,20,65,29,59,15,42};
    // wm.print_board();
    // for (unsigned int i=0;i<temp.size();i++){
    //     auto move = wm.get_move_from_index(temp[i]);
    //     std::cout << move.first << "," << move.second << std::endl;
    //     wm.execute_move(move);
    //     wm.print_board();
    // }

//    WMChess::move_type move =  wm.get_move_from_index(10);
//    std::cout << move.first << "," << move.second << std::endl;

//    int move =  wm.get_move_from_string("20_16");
//    std::cout << move << std::endl;

    // std::vector<int> temp = wm.get_legal_moves();
    // for (unsigned int i=0;i<temp.size();i++){
    //     std::cout << "index = " << i << ", value = " << temp[i] << std::endl;

    // }

    // two_dimension_vector_print(wm.get_board());


   return 0;
}