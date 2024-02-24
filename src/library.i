%module(threads="1") library

%{
#include "wm_chess.h"
#include "libtorch.h"
#include "mcts.h"
%}

%include "std_vector.i"
%include "std_pair.i"
namespace std {
  %template(IntVector) vector<int>;
  %template(IntVectorVector) vector<vector<int>>;
  %template(DoubleVector) vector<double>;
  %template(DoubleVectorVector) vector<vector<double>>;
  %template(PairIntInt) pair<int,int>;
}

%include "std_string.i"

%include "wm_chess.h"
%include "mcts.h"

class NeuralNetwork {
 public:
  NeuralNetwork(std::string model_path, bool use_gpu, unsigned int batch_size);
  ~NeuralNetwork();
  void set_batch_size(unsigned int batch_size);
};
