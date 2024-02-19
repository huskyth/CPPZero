#include "wm_chess.h"

WMChess::WMChess(unsigned int n, int first_color)
    : n(n), cur_color(first_color), last_move(std::make_pair(-1, -1)){
  this->board = std::vector<int>(n, 0);
}
//TODO://board的网络形式需要再修改

// TODO:这个move需要经过转化
void WMChess::execute_move(move_type move) {
    int from = move.first;
    int to = move.second;
        assert(this->board[from] == this->cur_color);
        assert(this->board[to] == 0);
        // assert self.distance[from_int][to_int] == 1
        // self.pointStatus[from_int] = 0
        // self.pointStatus[to_int] = color
        // bake_point_status = copy.deepcopy(self.pointStatus)
        // self.pointStatus = shiftOutChessman(
        //     bake_point_status, self.distance)
}