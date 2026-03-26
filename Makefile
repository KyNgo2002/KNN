CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++20 -Iinclude

SRC = $(wildcard src/*.cpp)

TARGET = main

all: 
	g++ ${SRC} -Iinclude -o out

clean:
	rm out
