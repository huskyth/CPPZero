#pragma once
#include <vector>
#include <map>
#include<assert.h>
#include "common.h"
#include <algorithm>

class WMChess {
public:
  const int WHITE = -1;
  const int BLACK = 1;

  using move_type=std::pair<int, int>;
  using board_type = std::vector<int>;
  WMChess(unsigned int n, int first_color);
  inline unsigned int get_n() const { return this->n; }
  inline board_type get_board() const { return this->board; }
  inline move_type get_last_move() const { return this->last_move; }
  inline int get_current_color() const { return this->cur_color; }
  inline std::vector<std::vector<int>> get_distance(){ 
    if (this -> distance.size() == 0){
        this->distance = read_from_file("/Users/husky/CPPZero/src/distance.txt");
    }
    return this->distance; 
    }
    inline void init_move_index_tuple(){
        int index = 0;
        for (unsigned int from_point=0;from_point < 21;from_point++){
            std::vector<int> to_point_list = this->getNeighboors(from_point);
            sort(to_point_list.begin(), to_point_list.end());
            for(unsigned int i = 0;i < to_point_list.size();i++){
                int to_point = to_point_list[i];
                std::string temp = std::to_string(from_point) + "_" + std::to_string(to_point);
                this->move_string_index[temp] = index;
                this->move_index_tuple[index++] = std::make_pair(from_point, to_point);
            }
        }
    }
  inline int get_move_from_string(std::string move) {
    if(this->move_string_index.size()==0){
        this->init_move_index_tuple();
    }
    return this->move_string_index[move]; 
  }

  inline move_type get_move_from_index(int id) {
    if(this->move_index_tuple.size()==0){
        this->init_move_index_tuple();
    }
    return this->move_index_tuple[id]; 
  }

  inline void print_board() const {  
    for(unsigned int i=0;i < this->board.size();i++){
        std :: cout << this->board[i] << " ";
    }
    std :: cout << std::endl;
  }

  void execute_move(move_type move);
  std::vector<int> get_game_status();
  void set_board(board_type board);
  std::vector<int> get_legal_moves();

private:

unsigned int n;
board_type board;
move_type last_move;
int cur_color;
std::vector<std::vector<int>> distance;
std::map<int,move_type> move_index_tuple;
std::map<std::string,int> move_string_index;
board_type shiftOutChessman(board_type board);
bool check(int chessman, WMChess::board_type pointStatus, std::vector<int> checkedChessmen);
std::vector<int> getNeighboors(int chessman);
};