#coding = utf-8
'''
现在有一个C++工程，包括多个.cpp和.h文件，其中，假设example.cc是预备暴露给python的接口源文件（example.i对应的函数实现源文件），但是其实现使通过调用other-cpp-wrapper.cc里面的方法，而example_wrap.cxx是SWIG根据example.i生成的包装源文件并且使用了外部的.a 静态库，也就是说example_wrap.cxx并不包含也不知道other-cpp-wrapper.cc里面的任何方法实现。
'''
from distutils.core import setup, Extension
example_module = Extension('_library', # 模块名称，和example.i中的%module对应加下划线
                extra_compile_args = ['-std=c++17'],
            include_dirs = ['/Users/husky/Downloads/libtorcharm64/include'],
            library_dirs = [],
            libraries = [],
            sources = ['library_wrap.cxx','common.cpp', 'libtorch.cpp', 'mcts.cpp', 'wm_chess.cpp'],
            extra_objects = []
            )
setup(name = 'library', #定义模块基本信息
      version = '0.1',
      author = 'husky',
      description = """Simple libtorch demo""",
      ext_modules = [example_module],
      py_modules = ["ok"],
      ) 
