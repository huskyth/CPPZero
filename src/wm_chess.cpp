#include "wm_chess.h"

WMChess::WMChess(unsigned int n, int first_color)
    : n(n), cur_color(first_color), last_move(std::make_pair(-1, -1)){
  this->board = {1, 1, 1, 1, 1, 0, 0, -1, 1, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0};
}
//TODO:board的网络形式需要再修改

// TODO:这个move需要经过转化
void WMChess::execute_move(move_type move) {
    int from = move.first;
    int to = move.second;
    assert(this->board[from] == this->cur_color);
    assert(this->board[to] == 0);
    assert(this->get_distance()[from][to] == 1);
    this->board[from] = 0;
    this->board[to] = this->cur_color;
    this->last_move = std::make_pair(from, to);
    board_type bake_point_status = std::vector<int>(0,0);
    for (unsigned int i=0;i < this->board.size();i++){
        bake_point_status.push_back(this->board[i]);
    }
    this->board = shiftOutChessman(
        bake_point_status);
    this->cur_color = -this->cur_color;
}

std::vector<int> WMChess::getNeighboors(int chessman){
    std::vector<int> neighboorChessmen = std::vector<int>(0,0);
    for (unsigned int eachChessman = 0;eachChessman < this->get_distance()[chessman].size();eachChessman++){
        int eachDistance = distance[chessman][eachChessman];
        if(eachDistance == 1){
            neighboorChessmen.push_back(eachChessman);
        }
    }
    return neighboorChessmen;
}

bool WMChess::check(int chessman, board_type pointStatus, std::vector<int> checkedChessmen){

    checkedChessmen.push_back(chessman);
    bool dead = true;
    std::vector<int> neighboorChessmen = getNeighboors(chessman);
    for(unsigned int i = 0;i < neighboorChessmen.size();i++){
        int neighboorChessman = neighboorChessmen[i];
        if (!count(checkedChessmen.begin(),checkedChessmen.end(), neighboorChessman)){
            if (pointStatus[neighboorChessman] == pointStatus[chessman]){
                dead = check(neighboorChessman, pointStatus, checkedChessmen);
                if (!dead){
                    return dead;
                }
            }else if(pointStatus[neighboorChessman] == 0){
                return false;
            }
        }
    }

    return dead;


}

WMChess::board_type WMChess::shiftOutChessman(board_type board) {
    std::vector<int> deadChessmen = std::vector<int>(0,0);
    board_type bakPointStatus = std::vector<int>(0,0);
    for (unsigned int i=0;i < board.size();i++){
        bakPointStatus.push_back(board[i]);
    }
    for (unsigned int chessman=0;chessman < board.size();chessman++){
        std::vector<int> checkedChessmen = std::vector<int>(0,0);
        bool dead = true;
        int color = board[chessman];
        if (color != 0){
            dead = check(chessman, board, checkedChessmen);
        }
        if(dead){
            deadChessmen.push_back(chessman);
        }
        board = bakPointStatus;
    }

    for (unsigned int j=0;j < deadChessmen.size();j++){
        board[deadChessmen[j]] = 0;
    }
    return board;
}