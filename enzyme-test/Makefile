build:
	clang++ enzyme-test.cpp -S -emit-llvm -o input.ll -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -std=c++11
	/opt/homebrew/Cellar/llvm/13.0.0_2/bin/opt input.ll -enable-new-pm=0 -load=/opt/homebrew/Cellar/enzyme/0.0.26/lib/LLVMEnzyme-13.dylib -enzyme -o output.ll -S
