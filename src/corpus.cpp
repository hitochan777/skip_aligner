#include "corpus.h"

using namespace std;

Dict::Dict() : b0_("<bad0>") {
	words_.reserve(1000);
}

unsigned Dict::max() const {
	return words_.size();
}

bool Dict::is_ws(char x) {
	return (x == ' ' || x == '\t');
}

void Dict::ConvertWhitespaceDelimitedLine(const string& line, WordVector* out) {
	size_t cur = 0;
	size_t last = 0;
	int state = 0;
	out->clear();
	while(cur < line.size()) {
		if (is_ws(line[cur++])) {
			if (state == 0) continue;
			out->push_back(Convert(line.substr(last, cur - last - 1)));
			state = 0;
		} else {
			if (state == 1) continue;
			last = cur - 1;
			state = 1;
		}
	}
	if (state == 1)
		out->push_back(Convert(line.substr(last, cur - last)));
}

WordID Dict::Convert(const string& word, bool frozen) {
	Map::iterator i = d_.find(word);
	if (i == d_.end()) {
		if (frozen){
			return 0;
		}
		words_.push_back(word);
		d_[word] = words_.size();
		return words_.size();
	}
	else {
		return i->second;
	}
}

const string& Dict::Convert(const WordID id) const {
	if (id == 0){
		return b0_;
	}
	return words_[id-1];
}

const string Dict::Convert(const WordVector wv) const {
	string str="";
	for(unsigned int i = 0;i<wv.size();++i){
		str+=Convert(wv[i])+" ";
	}
	return str;
}

void ReadFromFile(const string& filename,Dict* d,vector<vector<unsigned> >* src,set<unsigned>* src_vocab) {
	src->clear();
	cerr << "Reading from " << filename << endl;
	ifstream in(filename.c_str());
	assert(in);
	string line;
	int lc = 0;
	while(getline(in, line)) {
		++lc;
		src->push_back(vector<unsigned>());
		d->ConvertWhitespaceDelimitedLine(line, &src->back());
		for (unsigned i = 0; i < src->back().size(); ++i){
			src_vocab->insert(src->back()[i]);
		}
	}
}
