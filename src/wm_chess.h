#pragma once
#include <vector>
#include<assert.h>
#include "common.h"
#include <algorithm>

class WMChess {
public:
  using move_type=std::pair<int, int>;
  using board_type = std::vector<int>;
  WMChess(unsigned int n, int first_color);
  inline unsigned int get_n() const { return this->n; }
  inline board_type get_board() const { return this->board; }

  inline move_type get_last_move() const { return this->last_move; }
  inline int get_current_color() const { return this->cur_color; }
  inline std::vector<std::vector<int>> get_distance(){ 
    if (this -> distance.size() == 0){
        std::cout << "distance null" << std::endl;
        this->distance = read_from_file("/Users/husky/CPPZero/src/distance.txt");
    }
    return this->distance; 
    }

  void execute_move(move_type move);

private:

unsigned int n;
board_type board;
move_type last_move;
int cur_color;
std::vector<std::vector<int>> distance;
board_type shiftOutChessman(board_type board);
bool check(int chessman, WMChess::board_type pointStatus, std::vector<int> checkedChessmen);
std::vector<int> getNeighboors(int chessman);
};