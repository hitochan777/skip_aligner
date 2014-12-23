#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <utility>
#include <vector>
#include <string> 
#include <array>
#include <algorithm>
#include <set>
#include <unordered_map>
#include <exception>
#include "MultiDimensionalArray.h"

#include <boost/functional/hash.hpp>

enum Smoothing {NO,VB,KN};

using namespace std;

//http://stackoverflow.com/questions/10405030/c-unordered-map-fail-when-used-with-a-vector-as-key
template <typename Container> // we can make this generic for any container [1]
struct container_hash {
	size_t operator()(Container const& c) const {
		return boost::hash_range(c.begin(), c.end());
	}
};

/*****************************typedef********************************/
typedef unsigned WordID;
typedef vector<WordID> WordVector;
typedef unordered_map<WordID, double> Word2Double;
typedef vector<Word2Double> Word2Word2Double;
typedef unordered_map<WordVector,Word2Double,container_hash<WordVector> > WordVector2Word2Double;
typedef vector<WordVector2Word2Double> VWV2WD;
typedef	unordered_map<WordVector,double,container_hash<WordVector> > WV2D;
typedef MultiDimensionalArray<double> MdDouble;
/********************************************************************/
extern const WordID kNULL;
extern const WordID SOS;
extern const WordID EOS;

#endif

