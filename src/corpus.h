#ifndef CPYPDICT_H_
#define CPYPDICT_H_

#include "common.h"

using namespace std;

class Dict {
	typedef unordered_map<string, unsigned, std::hash<string> > Map;

	private:
	string b0_;
	vector<string> words_;
	Map d_;

	public:
	Dict();
	unsigned max() const;
	static bool is_ws(char x);
	void ConvertWhitespaceDelimitedLine(const string& line, WordVector* out);
	WordID Convert(const string& word, bool frozen = false);
	const string& Convert(const WordID id) const;
	const string Convert(const WordVector wv) const;
};

void ReadFromFile(const string& filename,Dict* d,vector<vector<unsigned> >* src,set<unsigned>* src_vocab);

#endif
