#pragma once
#include <vector>
#include<assert.h>

class WMChess {
public:
  using move_type=std::pair<int, int>;
  using board_type = std::vector<int>;
  WMChess(unsigned int n, int first_color);
  inline unsigned int get_n() const { return this->n; }
  inline board_type get_board() const { return this->board; }

  // TODO: last_move需要加在execute里面
  inline move_type get_last_move() const { return this->last_move; }
  inline int get_current_color() const { return this->cur_color; }

  void execute_move(move_type move);

private:

unsigned int n;
board_type board;
move_type last_move;
int cur_color;

};
