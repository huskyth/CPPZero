#include <libtorch.h>

#include <iostream>
#include "wm_chess.h"

int main() {
  WMChess wm_chess(7, 1);
  std::vector<int> temp = {1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  wm_chess.set_board(temp);
  std::cout << wm_chess.get_origin_board() << std:: endl;


  NeuralNetwork nn("../test/models/checkpoint.pt", false, 1);
  torch::Tensor v = nn.get_value(&wm_chess);
  
  std::cout << v << std::endl;
}
