#ifndef LM_H
#define LM_H
#include "common.h"
#include "ZMap.h"
#include "Vocab.h"
#include "utils.h"
#include "FracType.h"

#define MAX_DISCOUNT_ORDER 5

extern int _debug;

struct Bundle{
	double prob;
	FracType type;	
};

typedef struct Bundle Bundle;
typedef unordered_map<WordID,Bundle> BundleMap;

class Node {
	public:
		void clear(){
			bow=0;
			// probs.clear();
			childs.clear();
			bundle.clear();
		}
		Node(){
			bow=0;
			// probs.clear();
			childs.clear();
			bundle.clear();
		}
		~Node(){
			// probs.clear();
			// types.clear();
			bundle.clear();
			childs.clear();
		}
		void erase(WordID wrd);
		void prune(double threshold);
		void normalize();
		double bow;
		// ZMap<WordID, double> probs;
		// ZMap<WordID, FracType> types;
		BundleMap bundle;
		ZMap<WordID, Node> childs;
};

class LMStat {
	public:
		LMStat(int order){
			ngrams.clear();
			nwrds=0;
			for(int i=0;i<=order;i++){
				ngrams.push_back(0);
			}
		}
		void print(ostream& os){
			for(int i=0;i<(int)ngrams.size();i++){
				os<<i<<"-gram:"<<ngrams[i]<<endl;
			}
		}
		WordVector ngrams;
		int nwrds;
};

enum DiscountType{D_AD,D_WB};

void calculateDiscount(vector<double>& coc, vector<double>& discount);

class LM{
	public:
		Node* setNode(WordID* context, int clen);
		Node* lookupNode(WordID* context, int clen);
		double& setProb(WordID wrd, WordID* context, int clen);
		void setOrder(int order);
		void init(int order){
			setOrder(order);
			_vocab.add(unk);
		}
		void readCountFile(string filename);
		int readPlainFile(string filename, int order, double* p_weight, double threshold=0, double floor=0, int nSentence=-1);
		void readCountFile(string filename, unordered_map<string,FracType>& ngrams);
		void readPlainFile(string filename, double*& p_weight, unordered_map<string,FracType>& ngrams);
		bool addOneLine(string& curline, int order, double*& p_weight, double threshold, double floor);
		void addNgrams(unordered_map<string,pair<FracType,double>>& ngrams);
		void addNgram(WordVector& context, WordID wrd, double count);

		void reComputeLowerCount();
		void computeBOW(bool interpolate);
		void setSomeDetails();

		void knEstimate(bool interpolate, double* cutoff=NULL);
		void knEstimate(bool isBackward, bool interpolate, DiscountType dt, double* cutoff);
		void normalize();
		double& setBackoff(WordID* context, int clen);
		double probBO(WordID wrd, WordID* context, int clen);
		bool checkEntry(WordID wrd, WordID* context, int clen);
		void stat(WordID wrd, WordID* context, int clen, LMStat& lmstat);

		double sentLogProb(WordID* context, int slen, int order, LMStat* pStat);
		double sentLogProb(string& curline, int ord, LMStat* pStat=NULL);
		void read(string filename);
		void write(string filename);
		void writeCount(string filename);
		double ppl(string filename, LMStat* pStat, string probFilename="", bool average=true);
		void setKNcounts();
		void setKNcountsTam(double threshold);
		void calculateDiscounts(vector<vector<double> >& discounts, bool logrized=false);
		void adEstimate(Node& node, vector<vector<double> >& discounts, int order);
		void prune(Node& node, double* thresholds, int order);
		void prune(double* thresholds){
			if(thresholds!=NULL){
				prune(_root,thresholds,0);
			}
		}
		int order(){
			return (int)_numofGrams.size();
		}
		void clear(){
			_numofGrams.clear();
			_root.clear();
			_vocab.clear();
		}
		void cleanBadEntries(){
			vector<double> t; 
			for(int i=0;i<order();i++){
				t.push_back(0);
				prune(&t[0]);
			}
		}
		void setForAlignment(){
			_numofGrams.clear();
			_numofGrams.push_back(1);
			_numofGrams.push_back(1);
		}
		void setForAlignment(int n){//this is added line
			_numofGrams.clear();
			_numofGrams.resize(n);	
		}
		void addBigram(WordID wrd, WordID context, double count){
			if(count>1E-10){
				addFracCount(wrd, &context, 1, count, count);
			}
		}
		double bigramProb(WordID wrd, WordID context){
			return exp10(probBO(wrd,&context,1));
		}

		double ngramProb(WordID wrd, WordVector context){
			return exp10(probBO(wrd,&context[0],(int)context.size()));	
		}

		void resetVocab();
		void copyDiscountedProb(VWV2WD& ttables,bool copyAllProb = false);
		void copyBOW(WV2D& bow);

	private:
		void calcNumOfGrams(){
			_numofGrams.clear();
			calcNumOfGrams(_root,0);
		}
		Node* addFracCount(WordID wrd, WordID* context, int clen, double fcount, double ftype);
		Node* addFracCount(WordID wrd, WordID* context, int clen, double fcount, FracType& ftype);


		void computeBOWByLayer(Node& node, WordVector& context, int order, bool interpolate);
		void calcNumOfGrams(Node& node, int order);
		int readNumofGrams(ifstream& is);
		void readNgram(ifstream& is, int order);
		void print(ostream& os, Node& node, string sufix, WordVector& context, int order, bool countOnly);
		void wbEstimate(Node& node);
		void reComputeLowerCount(Node& node, WordVector& context, int order);
		void clearCountsByLayer(Node& node, int order, int exception);
		void collectCountofCount(Node& node, vector<double>& coc, int order,bool logrized);
		void setKNcountsByLayer(Node& node, WordVector& context, int order);
		void setKNcountsByLayerTam(Node& node, WordVector& context, int order, double threshold);
		void setMissKNcountsByLayer(Node& node, WordVector& context, int order);
		void cleanBadEntriesByLayer(Node& node, WordVector& context, int order);
		void _copyDiscountedProb(Node& node,WordVector& context,VWV2WD& ttables,bool copyAllProb);
		void _copyBOW(Node& node,WordVector& context,WV2D& bow);

		vector<int> _numofGrams;
		Node _root;
		Vocab _vocab;
};
void readPlainFile(string filename, int order, double*& p_weight, int nSentence,unordered_map<string,pair<FracType,double>>& ngrams);

#endif
