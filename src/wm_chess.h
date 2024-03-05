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
  std::map<unsigned int, std::pair<int,int>> ARRAY_TO_IMAGE {
    {0, std::make_pair(0, 3)}, {15, std::make_pair(6, 3)}, {6, std::make_pair(3, 0)}, {10, std::make_pair(3, 6)},
    {1, std::make_pair(0, 2)}, {3, std::make_pair(0, 4)}, {2, std::make_pair(1, 3)},
    {4, std::make_pair(2, 0)}, {7, std::make_pair(4, 0)}, {5, std::make_pair(3, 1)},
    {8, std::make_pair(2, 6)}, {9, std::make_pair(3, 5)}, {11, std::make_pair(4, 6)},
    {12, std::make_pair(5, 3)}, {13, std::make_pair(6, 2)}, {14, std::make_pair(6, 4)},
    {20,std::make_pair(3, 3)},
    {16,std::make_pair(2, 3)}, {17, std::make_pair(3, 2)}, {18, std::make_pair(3, 4)}, {19, std::make_pair(4, 3)}
};

  using move_type=std::pair<int, int>;
  using board_type = std::vector<int>;
  using input_board_type = std::vector<std::vector<int>>;
  WMChess(unsigned int n, int first_color);
  inline unsigned int get_n() const { return this->n; }
  inline input_board_type get_board() { return this->transfer(); }
  inline board_type get_origin_board() { return this->board; }
  inline move_type get_last_move() const { return this->last_move; }
  inline int get_current_color() const { return this->cur_color; }
  inline unsigned int get_action_size() const { return 72; }

  inline std::vector<std::vector<int>> get_distance(){ 
    if (this -> distance.size() == 0){
        this->distance = read_from_file("../src/distance.txt");
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

  inline int get_move_from_move(move_type move) {
    std::string temp = std::to_string(move.first) + "_" + std::to_string(move.second);
    if(this->move_string_index.size()==0){
        this->init_move_index_tuple();
    }
    if(this->move_string_index.count(temp) ==0){
      assert(temp == "-1_-1");
      return -1;
    }
    return this->move_string_index[temp]; 
  }

  inline move_type get_move_from_index(int id) {
    if(this->move_index_tuple.size()==0){
        this->init_move_index_tuple();
    }
    if(this->move_index_tuple.count(id) ==0){
      std::cout << "no key return (-1,-1) in get_move_from_index" << std::endl;
      return move_type(-1,-1);
    }
    return this->move_index_tuple[id]; 
  }

  inline void print_origin_board() const {  
    for(unsigned int i=0;i < this->board.size();i++){
      if(i<10) std :: cout << i << "  ";
      else{
          std :: cout << i << " ";
      }
    }
    std::cout << std::endl;
    for(unsigned int i=0;i < this->board.size();i++){
      if(this->board[i] >= 0)
        std :: cout << this->board[i] << "  ";
      else{
        std :: cout << this->board[i] << " ";
      }
    }
    std :: cout << std::endl;
  }

  void execute_move(move_type move);
  std::vector<int> get_game_status();
  void set_board(board_type board);
  std::vector<int> get_legal_moves();
  input_board_type transfer();
  std::pair<int,int> find_row_column_in_map(move_type move);

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