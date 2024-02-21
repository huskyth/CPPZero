#include <iostream>
#include <assert.h>

#include "wm_chess.h"

int main(){
    WMChess wm(7, 1);
    // std::cout << "from " << wm.get_last_move().first << " to " << wm.get_last_move().second << std::endl;

    // wm.get_distance();
    std::vector<int> temp = {27,42,15,49,20,65,29,59,15,42};
    wm.print_board();
    for (unsigned int i=0;i<temp.size();i++){
        auto move = wm.get_move_from_index(temp[i]);
        std::cout << move.first << "," << move.second << std::endl;
        wm.execute_move(move);
        wm.print_board();
    }

//    WMChess::move_type move =  wm.get_move_from_index(10);
//    std::cout << move.first << "," << move.second << std::endl;




    return 0;
}