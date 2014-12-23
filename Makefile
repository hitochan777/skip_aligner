CC=g++
CFLAGS=-c -Wall -std=c++0x
LDFLAGS=
LIBRARY=-lboost_program_options
SRCDIR=src
SOURCES=$(SRCDIR)/FracType.cpp  $(SRCDIR)/LM.cpp  $(SRCDIR)/LogDouble.cpp  $(SRCDIR)/Vocab.cpp  $(SRCDIR)/corpus.cpp  $(SRCDIR)/fast_align.cpp  $(SRCDIR)/ttables.cpp  $(SRCDIR)/utils.cpp
OBJECTS=$(SOURCES:.cpp=.o)
	EXECUTABLE=skip_aligner

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBRARY)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(SRCDIR)/*.o $(EXECUTABLE)
