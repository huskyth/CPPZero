
#include "common.h"
void two_dimension_vector_print(std::vector<std::vector<int>> printed){
    unsigned column = printed[0].size();
    unsigned row = printed.size();
                            std::cout << "  ";
    for (unsigned int i = 0; i < column; i++) {
                        std::cout << i + 1 << " ";
    }
    std::cout << std::endl;


      for (unsigned int i = 0; i < printed.size(); i++) {
                                std::cout << i + 1 << " ";
      for (unsigned int j = 0; j < printed[i].size(); j++) {
                std::cout << printed[i][j] << " ";

      }
      std::cout << std::endl;
      }

}

void one_dimension_vector_print(std::vector<double> printed){
    unsigned column = printed.size();
      for (unsigned int i = 0; i < printed.size(); i++) {
        std::cout << printed[i] << " ";
      }

}
std::vector<std::vector<int>> transfer(std::string x){
    unsigned n = x.length();

    std::vector<std::vector<int>> result = std::vector<std::vector<int>>(0, std::vector<int>(0, 0));
    std::vector<int> temp_vector;
    for (unsigned int i =1;i < n-1;i++){
        if (x[i] == ' '){
        }else if(x[i] == ','){
        }else if(x[i] == '0' || x[i] == '1'){
            unsigned int temp = x[i] - '0';
            temp_vector.push_back(temp);
        }else if(x[i] == '['){
            temp_vector = std::vector<int>(0,0);
        }else if(x[i] == ']'){
            result.push_back(temp_vector);
        }


    }
    return result;

}
std::vector<std::vector<int>>  read_from_file(std::string path)
{
	std::ifstream srcFile(path, std::ios::in); //以文本模式打开txt文件
	std::string x;
	while (getline(srcFile, x))
	{ 
	}
    auto result = transfer(x);
	srcFile.close();
    return result;
}
