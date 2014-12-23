/*
 * MultiDimensionalArray.h
 *
 *  Created on: 2014/12/12
 *      Author: otsuki
 */

#ifndef MULTIDIMENSIONALARRAY_H_
#define MULTIDIMENSIONALARRAY_H_

#include <vector>
#include <cstdarg>
#include <sstream>
#include <stdexcept>

using namespace std;

template<class T> /*d: dimension*/
class MultiDimensionalArray {
private:
	vector<unsigned int> __sizes;
	vector<T> __v;
	vector<unsigned int> __block;
public:
	MultiDimensionalArray();
	MultiDimensionalArray(int dimension, ...);
	virtual ~MultiDimensionalArray();
	T operator()(unsigned int n, ...) const;
	T& operator()(unsigned int n, ...);
	MultiDimensionalArray operator+(const MultiDimensionalArray<T>& p2);
	MultiDimensionalArray operator*(const MultiDimensionalArray<T>& p2);
	MultiDimensionalArray operator/(const MultiDimensionalArray<T>& p2);
	MultiDimensionalArray operator-(const MultiDimensionalArray<T>& p2);

	void operator+=(const MultiDimensionalArray<T>& p2);
	void operator*=(const MultiDimensionalArray<T>& p2);
	void operator/=(const MultiDimensionalArray<T>& p2);
	void operator-=(const MultiDimensionalArray<T>& p2);
	unsigned int size() const;
	void resize(int dimension, ...);
	string toString() const;
	void clear();
};

template<class T>
MultiDimensionalArray<T>::MultiDimensionalArray(){}

template<class T>
MultiDimensionalArray<T>::MultiDimensionalArray(int dimension, ...) {
	va_list ap;
	unsigned int j;
	unsigned int tmp = 1;
	va_start(ap, dimension); //Requires the last fixed parameter (to get the address)
	for (j = 0; j < dimension; j++) {
		unsigned int s = va_arg(ap, unsigned int);
		__sizes.push_back(s); //Requires the type to cast to. Increments ap to the next argument.
		tmp *= s;
	}
	__v.resize(tmp, 0);
	for (j = 0; j < dimension; j++) {
		tmp /= __sizes[j];
		__block.push_back(tmp);
	}
	va_end(ap);
}

template<class T>
MultiDimensionalArray<T>::~MultiDimensionalArray() {
}

template<class T>
void MultiDimensionalArray<T>::resize(int dimension, ...) {
	va_list ap;
	unsigned int j;
	unsigned int tmp = 1;
	vector<T>().swap(__v);
	vector<unsigned int>().swap(__sizes);
	vector<unsigned int>().swap(__block);
	va_start(ap, dimension); //Requires the last fixed parameter (to get the address)
	for (j = 0; j < dimension; j++) {
		unsigned int s = va_arg(ap, unsigned int);
		__sizes.push_back(s); //Requires the type to cast to. Increments ap to the next argument.
		tmp *= s;
	}
	__v.resize(tmp, 0);
	for (j = 0; j < dimension; j++) {
		tmp /= __sizes[j];
		__block.push_back(tmp);
	}
	va_end(ap);
}

template<class T>
void MultiDimensionalArray<T>::clear() {
	__v.clear();
	vector<T>().swap(__v);
	return;
}

template<class T>
MultiDimensionalArray<T> MultiDimensionalArray<T>::operator+(
		const MultiDimensionalArray<T>& p2) {
	throw "not implemented yet";
}

template<class T>
MultiDimensionalArray<T> MultiDimensionalArray<T>::operator-(
		const MultiDimensionalArray<T>& p2) {
	throw "not implemented yet";
}

template<class T>
MultiDimensionalArray<T> MultiDimensionalArray<T>::operator*(
		const MultiDimensionalArray<T>& p2) {
	throw "not implemented yet";
}

template<class T>
MultiDimensionalArray<T> MultiDimensionalArray<T>::operator/(
		const MultiDimensionalArray<T>& p2) {
	throw "not implemented yet";
}

template<class T>
void MultiDimensionalArray<T>::operator+=(const MultiDimensionalArray<T>& p2) {
	if (this->size() != p2.size()) {
		throw "both array size must be the same";
	}
	for (unsigned int i = 0; i < __v.size(); ++i) {
		__v[i] += p2.__v[i];
	}
}

template<class T>
void MultiDimensionalArray<T>::operator-=(const MultiDimensionalArray<T>& p2) {
	if (this->size() != p2.size()) {
		throw "both array size must be the same";
	}
	for (unsigned int i = 0; i < __v.size(); ++i) {
		__v[i] -= p2.__v[i];
	}
}

template<class T>
void MultiDimensionalArray<T>::operator*=(const MultiDimensionalArray<T>& p2) {
	if (this->size() != p2.size()) {
		throw "both array size must be the same";
	}
	for (unsigned int i = 0; i < __v.size(); ++i) {
		__v[i] *= p2.__v[i];
	}
}

template<class T>
void MultiDimensionalArray<T>::operator/=(const MultiDimensionalArray<T>& p2) {
	if (this->size() != p2.size()) {
		throw "both array size must be the same";
	}
	for (unsigned int i = 0; i < __v.size(); ++i) {
		__v[i] /= p2.__v[i];
	}
}

template<class T>
unsigned int MultiDimensionalArray<T>::size() const {
	return __v.size();
}

template<class T>
T MultiDimensionalArray<T>::operator()(unsigned int n, ...) const {
	if (n != __sizes.size()) {
		throw "specified indices length not matching";
	}
	unsigned int index = 0;
	va_list ap;
	va_start(ap, n); //Requires the last fixed parameter (to get the address)
	for (unsigned int j = 0; j < n; j++) {
		unsigned int i = va_arg(ap, unsigned int);
		if (i >= __sizes[j]) {
			cerr<<"i="<<i<<",__sizes[j]="<<__sizes[j]<<endl;
			throw out_of_range("index out of range");
		}
		index += __block[j] * i;
	}
	va_end(ap);
	return __v[index];
}

template<class T>
T& MultiDimensionalArray<T>::operator()(unsigned int n, ...) {
	if (n != __sizes.size()) {
		throw "specified indices length not matching";
	}
	unsigned int index = 0;
	va_list ap;
	va_start(ap, n);
	for (unsigned int j = 0; j < n; ++j) {
		unsigned int i = va_arg(ap, unsigned int);
		if (i >= __sizes[j]) {
			cerr<<"i="<<i<<",__sizes[j]="<<__sizes[j]<<endl;
			throw out_of_range("index out of range");
		}
		index += __block[j] * i;
	}
	va_end(ap);
	return __v[index];
}

template<class T>
string MultiDimensionalArray<T>::toString() const {
	stringstream ss;
	for (unsigned int i = 0; i < __v.size(); ++i) {
		ss << __v[i] << " ";
	}
	return ss.str();
}

#endif /* MULTIDIMENSIONALARRAY_H_ */
