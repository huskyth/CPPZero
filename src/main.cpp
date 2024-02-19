#include <iostream>
#include <assert.h>

#include "wm_chess.h"
#include "common.h"

int main(){
    WMChess wm(7, 1);
    std::cout << "from " << wm.get_last_move().first << " to " << wm.get_last_move().second << std::endl;

    std::string path = "/Users/husky/CPPZero/src/distance.txt";
    read_from_file(path);

    return 0;
}