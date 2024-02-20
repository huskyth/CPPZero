#include <iostream>
#include <assert.h>

#include "wm_chess.h"

int main(){
    WMChess wm(7, 1);
    std::cout << "from " << wm.get_last_move().first << " to " << wm.get_last_move().second << std::endl;

    wm.get_distance();

    return 0;
}