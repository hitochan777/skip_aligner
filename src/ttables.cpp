#include "ttables.h"
#include <new>

using namespace std;

TTable::TTable(int _n){
	this->n = _n;//length of vector e
	ttables.resize(_n+1);//index of ttable and counts shows the length of vector e
	counts.resize(_n+1);
}

double TTable::prob(const WordVector& e, const WordID& f){
	WordVector2Word2Double::const_iterator it = ttables[n].find(e);
	if(it!=ttables[n].end()){// if target n-gram exists
		const Word2Double& w2d = it->second;
		const Word2Double::const_iterator it = w2d.find(f);
		if(it==w2d.end()){// if source word does not exist
			return 1e-9;//to avoid zero frequency problem
		}
		return it->second;
	}
	else{// if target n-gram does NOT exists 
		return 1e-9;//to avoid zero frequency problem
	}
}

double TTable::backoffProb(const WordVector& e, const WordID& f){
	double p = 1.0;
	for(int i = 0; i<=n ;++i){
		WordVector2Word2Double::const_iterator it = ttables[n-i].find(WordVector(e.begin()+i,e.end()));
		if(it!=ttables[i].end()){// if target i-gram exists
			const Word2Double& w2d = it->second;
			const Word2Double::const_iterator it = w2d.find(f);
			if(it==w2d.end()){// if source word does not exist
				if(bow.find(e)!=bow.end()){
					p *= bow[e];
				}
				continue;
			}
			return p*it->second;
		}
	}
	return 1e-9;
}

void TTable::Increment(const WordVector& e, const int& f) {
	int len = e.size();
	if(len!=n){
		cerr<<"size is not correct in Increment"<<endl;	
		return ;
	}
	counts[n][e][f] += 1.0;
}

void TTable::Increment(const WordVector& e, const int& f, double x) {
	int len = e.size();
	if(len!=n){
		cerr<<"size is not correct in Increment"<<endl;	
		return ;
	}
	counts[n][e][f] += x;
}

void TTable::NormalizeVB(const double alpha) {
	ttables.swap(counts);
	for (WordVector2Word2Double::iterator it = ttables[n].begin(); it != ttables[n].end(); ++it) {
		double tot = 0;
		Word2Double& cpd = it->second;
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			tot += it2->second + alpha;
		}
		if (!tot){
			tot = 1;
		}
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			it2->second = exp(Md::digamma(it2->second + alpha) - Md::digamma(tot));
		}
	}
	counts.clear();
	VWV2WD().swap(counts);
	counts.resize(n+1);
}

void TTable::Normalize(bool lower) {
	ttables.swap(counts);
	for (WordVector2Word2Double::iterator it = ttables[n].begin(); it != ttables[n].end(); ++it) {
		double tot = 0;
		Word2Double& cpd = it->second;
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			if(lower){
				for(int i = 1;i <= n; ++i ){//calculate counts for 1~(n-1) grams from n-gram counts
					try{
						ttables[n-i][WordVector((it->first).begin()+i,(it->first).end())][it2->first]
							+= ttables[n][it->first][it2->first];
					}
					catch (bad_alloc& ba){
						cerr << "bad_alloc caught: " << ba.what() << '\n';
						cerr << "i="<<i<<" "<<"it->first.size()=="<<it->first.size()<<endl;
						return; 
					}	      
				}
			}
			tot += it2->second;
		}
		if (!tot){
			tot = 1;
		}
		for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
			it2->second /= tot;
		}
	}
	if(lower){
		for(int i = 0;i <= n - 1; ++i){//normalize probabilities for 1~(n-1) grams 
			for (WordVector2Word2Double::iterator it = ttables[i].begin(); it != ttables[i].end(); ++it) {
				double tot = 0;
				Word2Double& cpd = it->second;
				for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
					tot += it2->second;
				}
				if (!tot){
					tot = 1;
				}
				for (Word2Double::iterator it2 = cpd.begin(); it2 != cpd.end(); ++it2){
					it2->second /= tot;
				}
			}
		}
	}
	counts.clear();
	VWV2WD().swap(counts);
	counts.resize(n+1);
}

void TTable::copyFromKneserNeyLM(bool copyBOW, bool  copyAllProb){
	lm.setOrder(n+1);
	cerr<<"kn Estimate start"<<endl;
	lm.knEstimate(true);//interpolation is set to true.
	cerr<<"kn Estimate end"<<endl;
	//lm.write("/home/otsuki/Research/aligner/debug.txt");
	ttables.clear();
	VWV2WD().swap(ttables);
	ttables.resize(n+1);
	if(copyBOW){
		lm.copyBOW(this->bow);
	}
	lm.copyDiscountedProb(this->ttables,copyAllProb);
	cerr<<"copy done"<<endl;
	lm.clear();
	cerr<<"clear done"<<endl;	
	return ;
}

// adds counts from another TTable - probabilities remain unchanged
TTable& TTable::operator+=(const TTable& rhs) {
	if(rhs.n != n){
		std::cerr<<"Two tables have different n-gram number."<<std::endl;
	}
	for(int i = 0;i<std::min(rhs.n,n); ++i){
		for (WordVector2Word2Double::const_iterator it = rhs.counts[i].begin(); it != rhs.counts[i].end(); ++it) {
			const Word2Double& cpd = it->second;
			Word2Double& tgt = counts[i][it->first];
			for (Word2Double::const_iterator it2 =  cpd.begin();it2!=cpd.end();++it2){
				tgt[it2->first] += it2->second;
			}
		}
	}
	return *this;
}

void TTable::ShowCounts(int index,Dict& d) {
	_ShowCounts(index,d);
}

void TTable::ShowCounts(Dict& d) {
	_ShowCounts(n,d);
}

void TTable::ShowTTable(int index, Dict& d) {
	_ShowTTable(index,d);
}

void TTable::ShowTTable(Dict& d){
	_ShowTTable(n,d);
}



void TTable::_ShowCounts(int index, Dict& d) {
	for (WordVector2Word2Double::const_iterator it = counts[index].begin(); it != counts[index].end(); ++it) {
		const Word2Double& cpd = it->second;
		for (Word2Double::const_iterator it2 =  cpd.begin();it2!=cpd.end();++it2){
			cerr << "c(" << d.Convert(it2->first) << '|' << d.Convert(it->first) << ") = " << it2->second << endl;
		}
	}
}

void TTable::_ShowTTable(int index, Dict& d){
	fprintf(stderr,"showing %d-gram prob table\n",index);
	fprintf(stderr,"size: %lu\n",ttables[index-1].size());
	fprintf(stderr,"skipping cell with zero prob\n");
	for (WordVector2Word2Double::const_iterator it = ttables[index].begin(); it != ttables[index].end(); ++it) {
		const Word2Double& cpd = it->second;
		for (Word2Double::const_iterator it2 =  cpd.begin();it2!=cpd.end();++it2){
			if(it2->second<0.1){
			       	continue;//do not print prob with 0
			}
			fprintf(stderr,"Pr(%s|%s) = %lf\n", d.Convert(it2->first).c_str(), d.Convert(it->first).c_str(),it2->second);
		}
	}

}
