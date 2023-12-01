OBJS	= framework.o bunjee.o
SOURCE	= framework.cpp bunjee.cpp
HEADER	= eglew.h framework.h freeglut_ext.h freeglut_std.h freeglut_ucall.h freeglut.h glew.h glut.h glxew.h
OUT	= hazi3
CC	 = g++
FLAGS	 = -g -c -Wall
LFLAGS	 = -lglut -lGLU -lGL -lGLEW

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(LFLAGS)

framework.o: framework.cpp
	$(CC) $(FLAGS) framework.cpp 

.o: .cpp
	$(CC) $(FLAGS) .cpp 


clean:
	rm -f $(OBJS) $(OUT)