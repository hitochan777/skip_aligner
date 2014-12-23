#include "common.h"
#include "ttables.h"
#include "corpus.h"
#include "da.h"
#include "LM.h"

#include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
namespace po = boost::program_options;

struct PairHash {
	size_t operator()(const pair<short,short>& x) const {
		return (unsigned short)x.first << 16 | (unsigned)x.second;
	}
};
/**************declaration of global variable**************************
 * defined here so that printProcess() can use these.
 * 
 */
Dict d;// this variable is going to be used throughout this program
const WordID kNULL = d.Convert("<null>");
const WordID SOS = d.Convert("<s>");
const WordID EOS = d.Convert("</s>");
double likelihood = 0;
double base2_likelihood = 0;
double denom = 0.0;
int lc = 0;
bool flag = false;
double c0 = 0;
double emp_feat = 0;
int toks = 0;
unordered_map<pair<short, short>, unsigned,PairHash > size_counts;

/***********************************************************************/
void printVector(const vector<unsigned>& v){
	for(unsigned int i = 0;i<v.size();++i){
		cerr<<d.Convert(v[i])<<" ";
	}
	cerr<<endl<<flush;
	return ;
}

void ParseLine(const string& line,vector<unsigned>* src,vector<unsigned>* trg,bool is_reverse){
	static const unsigned kDIV = d.Convert("|||");
	static vector<unsigned> tmp;
	src->clear();
	trg->clear();
	d.ConvertWhitespaceDelimitedLine(line, &tmp);
	unsigned i = 0;
	if(is_reverse){
		src->push_back(SOS);
	}
	while(i < tmp.size() && tmp[i] != kDIV) {
		src->push_back(tmp[i]);
		++i;
	}
	if(is_reverse){
		src->push_back(EOS);
	}
	if(!is_reverse){
		trg->push_back(SOS);
	}
	if (i < tmp.size() && tmp[i] == kDIV) {
		++i;
		for (; i < tmp.size() ; ++i){
			trg->push_back(tmp[i]);
		}
	}
	if(!is_reverse){
		trg->push_back(EOS);
	}
	if(is_reverse){
		src->swap(*trg);
	}
}

void ParseLineFromSeparateFiles(const string& fline,const string& eline,vector<unsigned>* src,vector<unsigned>* trg,bool is_reverse){
	static vector<unsigned> tmp;
	src->clear();
	trg->clear();
	d.ConvertWhitespaceDelimitedLine(fline, &tmp);
	if(is_reverse){
		src->push_back(SOS);
	}
	for(unsigned i = 0;i < tmp.size();++i) {
		src->push_back(tmp[i]);
	}
	if(is_reverse){
		src->push_back(EOS);
	}
	tmp.clear();
	d.ConvertWhitespaceDelimitedLine(eline, &tmp);
	if(!is_reverse){
		trg->push_back(SOS);
	}
	for(unsigned i = 0;i < tmp.size();++i) {
		trg->push_back(tmp[i]);
	}
	if(!is_reverse){
		trg->push_back(EOS);
	}
	if(is_reverse){
		src->swap(*trg);
	}
	return ;
}

void printProcess(FILE* fp){
	fprintf(fp,"  log_e likelihood: %lf\n",likelihood);
	fprintf(fp,"  log_2 likelihood: %lf\n",base2_likelihood);
	fprintf(fp,"     cross entropy: %lf\n",(-base2_likelihood / denom));
	fprintf(fp,"        perplexity: %lf\n",pow(2.0, -base2_likelihood / denom));
	fprintf(fp,"      posterior p0: %lf\n",c0 / toks);
	fprintf(fp," posterior al-feat: %lf\n",emp_feat);
	//cerr << "     model tension: " << mod_feat / toks << endl;
	fprintf(fp,"       size counts: %lu\n",size_counts.size());
	return ;
}

bool verifyConf(po::variables_map* conf){
	return conf->count("help") ||
		(conf->count("input")!=0 && (conf->count("einput")!=0 || conf->count("finput")!=0)) ||
		(conf->count("input")==0 && conf->count("einput")==0 && conf->count("finput")==0) ||
		(conf->count("testset")!=0 && (conf->count("etestset")!=0 || conf->count("ftestset")!=0)) ||
		((conf->count("etestset")==0) ^ (conf->count("ftestset")==0));
}

bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
	po::options_description opts("Configuration options");
	opts.add_options()
		("input,i",po::value<string>(),"Parallel corpus input file")
		("finput",po::value<string>(),"source corpus input file")
		("einput",po::value<string>(),"target corpus input file")
		("reverse,r","Reverse estimation (swap source and target during training)")
		("iterations,I",po::value<unsigned>()->default_value(5),"Number of iterations of EM training")
		//("bidir,b", "Run bidirectional alignment")
		("favor_diagonal,d", "Use a static alignment distribution that assigns higher probabilities to alignments near the diagonal")
		("prob_align_null", po::value<double>()->default_value(0.08), "When --favor_diagonal is set, what's the probability of a null alignment?")
		("diagonal_tension,T", po::value<double>()->default_value(4.0), "How sharp or flat around the diagonal is the alignment distribution (<1 = flat >1 = sharp)")
		("train_line,l",po::value<int>()->default_value(-1),"how many lines to use when extracting sentences from training corpus.")
		("optimize_tension,o", "Optimize diagonal tension during EM")
		("variational_bayes,v","Infer VB estimate of parameters under a symmetric Dirichlet prior")
		("alpha,a", po::value<double>()->default_value(0.01), "Hyperparameter for optional Dirichlet prior")
		("no_null_word,N","Do not generate from a null token")
		("output_parameters,p", po::value<string>(), "Write model parameters to file")
		("beam_threshold,t",po::value<double>()->default_value(-4),"When writing parameters, log_10 of beam threshold for writing parameter (-10000 to include everything, 0 max parameter only)")
		("hide_training_alignments,H", "Hide training alignments (only useful if you want to use -x option and just compute testset statistics)")
		("testset,x", po::value<string>(), "After training completes, compute the log likelihood of this set of sentence pairs under the learned model")
		("ftestset", po::value<string>(), "Testset for source language. This must be paired with etestset and cannot be paired with testset.")
		("etestset", po::value<string>(), "Testset for target language. This must be paired with ftestset and cannot be paired with testset.")
		("test_align_output_file",po::value<string>(),"alignment result file for testdata")
		("train_align_output_file",po::value<string>(),"alignment result file for training data")
		("train_process_output_file",po::value<string>(),"ouput file for training process")
		("test_output_file",po::value<string>(),"output file for testdata except for alignment result")
		("no_add_viterbi,V","When writing model parameters, do not add Viterbi alignment points (may generate a grammar where some training sentence pairs are unreachable)")
		("force_align,f",po::value<string>(), "Load previously written parameters to 'force align' input. Set --diagonal_tension and --mean_srclen_multiplier as estimated during training.")
		("mean_srclen_multiplier,m",po::value<double>()->default_value(1), "When --force_align, use this source length multiplier")
		("window,w",po::value<int>()->default_value(5),"window size to use when using triplet or quadlet lexicon model")
		("smoothing,S",po::value<int>()->default_value(NO),"smoothing method: Maximum likelihood = 0, Variational Bayes = 1, Modified Kneser-ney = 2")
		("trigger",po::value<int>()->default_value(0),"Number of trigger, default to 0.")
		("with_null_trigger","If this is set, when target is null, triggers are allowed to be other values than null")
		("with_line_number","If this is set, line number will be displayed in each iteration")
		("show_ttable","whether to output ttable");
	po::options_description clo("Command line options");
	clo.add_options()
		("config", po::value<string>(), "Configuration file")
		("help,h", "Print this help message and exit");
	po::options_description dconfig_options, dcmdline_options;
	dconfig_options.add(opts);
	dcmdline_options.add(opts).add(clo);

	po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
	if (conf->count("config")) {
		ifstream config((*conf)["config"].as<string>().c_str());
		po::store(po::parse_config_file(config, dconfig_options), *conf);
	}
	po::notify(*conf);
	if (verifyConf(conf)){
		cerr << "Usage " << argv[0] << " [OPTIONS] -i corpus.fr-en\n";
		cerr << dcmdline_options << endl;
		return false;
	}
	return true;
}

int main(int argc, char** argv) {
	ios::sync_with_stdio(false);
	po::variables_map conf;
	if (!InitCommandLine(argc, argv, &conf)){
		return 1;
	}
	cout<<"hoge"<<endl;
	string testset, ftestset, etestset;

	if (conf.count("testset")){
		testset = conf["testset"].as<string>();
	}
	//actually this logic expression does not need two term. Only one is enough because verifyConf has verified both files are present in advance.
	else if(conf.count("ftestset") || conf.count("etestset")){	
		ftestset = conf["ftestset"].as<string>();
		etestset = conf["etestset"].as<string>();
	}

	const string fname = (conf.count("input")>0)?conf["input"].as<string>():"";
	const string ffname = (conf.count("finput")>0)?conf["finput"].as<string>():"";
	const string efname = (conf.count("einput")>0)?conf["einput"].as<string>():"";
	const bool is_reverse = conf.count("reverse") > 0;
	const int ITERATIONS = (conf.count("force_align")) ? 0 : conf["iterations"].as<unsigned>();
	//const double BEAM_THRESHOLD = pow(10.0, conf["beam_threshold"].as<double>());
	const bool use_null = (conf.count("no_null_word") == 0);

	const bool add_viterbi = (conf.count("no_add_viterbi") == 0);
	//const bool output_parameters = (conf.count("force_align")) ? false : conf.count("output_parameters");
	double diagonal_tension = conf["diagonal_tension"].as<double>();
	bool optimize_tension = conf.count("optimize_tension");
	bool hide_training_alignments = (conf.count("hide_training_alignments") > 0);
	const bool write_alignments = (conf.count("force_align")) ? true : !hide_training_alignments;
	const bool show_ttable = conf.count("show_ttable")>0;
	const bool ONE_INPUT_FILE = ffname.empty(); 
	const bool ONE_TEST_FILE = ftestset.empty();
	const bool DO_TEST = !testset.empty() || !ftestset.empty();
	const unsigned int WINDOW = conf["window"].as<int>();
	const int smooth = conf["smoothing"].as<int>();
	const int train_line = conf["train_line"].as<int>();
	if(conf.count("triplet")>0 && conf.count("quadlet")>0){
		cerr<<"You can't use both triplet and quadlet lexicon model. You can use EITHER of them."<<endl;
		return 0;
	}
	const bool WITH_NULL_TRIGGER = (conf.count("with_null_trigger") > 0);
	const bool WITH_LINE_NUMBER = (conf.count("with_line_number") > 0);
	const int TRIGGER = conf["trigger"].as<int>();
	if (conf.count("force_align")){
		testset = fname;
	}
	double prob_align_null = conf["prob_align_null"].as<double>();
	double prob_align_not_null = 1.0 - prob_align_null;
	const double alpha = conf["alpha"].as<double>();
	const bool favor_diagonal = conf.count("favor_diagonal");
	if (smooth == VB && alpha <= 0.0) {
		cerr << "--alpha must be > 0\n";
		return 1;
	}
	TTable t2s(TRIGGER+1);
	Word2Word2Double t2s_viterbi;
	double tot_len_ratio = 0;
	double mean_srclen_multiplier = 0;
	MdDouble probs;
	FILE *tof= (conf.count("train_process_output_file")>0)?fopen(conf["train_process_output_file"].as<string>().c_str(),"w"):stderr;
	FILE *taof  = (conf.count("train_align_output_file")>0)?fopen(conf["train_align_output_file"].as<string>().c_str(),"w"):stdout;
	/*******************startOfTraining************************/
	for (int iter = 0; iter < ITERATIONS; ++iter) {
		const bool final_iteration = (iter == (ITERATIONS - 1));
		const bool semi_final_iteration = (iter == (ITERATIONS - 2));
		ifstream in,fin,ein;
		if(ONE_INPUT_FILE){
			in.open(fname.c_str(), ifstream::in);
			if (!in) {
				cerr << "Can't read " << fname << endl;
				return 1;
			}
		}
		else{
			fin.open(ffname.c_str(), ifstream::in);
			ein.open(efname.c_str(), ifstream::in);
			if(!fin){
				cerr << "Can't read " << ffname << endl;
				return 1;
			}
			if(!ein){
				cerr << "Can't read " << efname << endl;
				return 1;
			}
		}
		if(tof!=stderr){
			fprintf(tof,"ITERATION %d%s\n",iter + 1,(final_iteration ? " (FINAL)" : ""));
		}
		fprintf(stderr,"ITERATION %d%s\n",iter + 1,(final_iteration ? " (FINAL)" : ""));
		likelihood = 0;
		denom = 0.0;
		lc = 0;
		flag = false;
		string line,fline,eline;
		vector<unsigned int> src, trg;
		c0 = 0;
		emp_feat = 0;
		toks = 0;
		while(true) {
			unsigned int I = 0;
			if(train_line != -1 && lc >= train_line){
				break;
			}
			if(ONE_INPUT_FILE){
				getline(in,line);
			}
			else{
				getline(fin,fline);
				getline(ein,eline);
			}
			if (ONE_INPUT_FILE && !in){
				break;
			}
			else if(!ONE_INPUT_FILE && (!fin || !ein)){
				break;
			}
			++lc;
			if(WITH_LINE_NUMBER){
				cout<<"line number:"<<lc<<endl;	
			}
			if (lc % 1000 == 0) { 
				fprintf(tof,".");
				if(tof!=stderr){
					fprintf(stderr,".");
				}
				flag = true;
			}
			if (lc %50000 == 0) { 
				fprintf(tof," [%d]\n",lc);
				if(tof!=stderr){
					fprintf(stderr," [%d]\n",lc);
				}
				flag = false;
			}
			src.clear();
			trg.clear();
			if(ONE_INPUT_FILE){
				ParseLine(line, &src, &trg,is_reverse);
			}
			else{
				ParseLineFromSeparateFiles(fline,eline,&src,&trg,is_reverse);
			}
			I = trg.size()-2;//excluding <s> and </s>
			if(src.size() == 0 || I == 0) {
				fprintf(stderr,"Error: %d\n%s\n",lc,line.c_str());
				return 1;
			}
			if(iter == 0){
				tot_len_ratio += static_cast<double>(src.size()) / static_cast<double>(I);
			}
			denom += src.size();
			switch(TRIGGER){
				case 0:
					probs.resize(TRIGGER+1,trg.size()+1);
					break;
				case 1:
					probs.resize(TRIGGER+1,trg.size()+1,trg.size()+1);
					break;
				case 2: 
					probs.resize(TRIGGER+1,trg.size()+1,trg.size()+1,trg.size()+1);
					break;
				default:
					throw runtime_error("invalid trigger");
					break;
			}
			if(iter == 0){
				++size_counts[make_pair<short,short>(src.size(),I)];
			}
			bool first_al = true;  // used when printing alignments
			toks += src.size();
			/*  0      1   2   3   4      I+1 I+2 */
			/* <NULL> <s> e_1 e_2 e_3 ... e_I </s> */
			/* trg[0] trg[1] trg[2] ... trg[I] trg[I+1] */
			/*   <s>    e_1   e_2   ...    e_I    </s>  */ 
			for (unsigned j = 0; j < src.size(); ++j){
				const unsigned& f_j = src[j];
				double sum = 0;
				double prob_a_j = 1.0 / (I + use_null); // uniform (model 1)
				if (use_null){
					if (favor_diagonal){
						prob_a_j = prob_align_null;
					}
					switch(TRIGGER){
						case 0:{
								   WordVector wv = {kNULL};
								   probs(1,0) = t2s.prob(wv, f_j) * prob_a_j;
								   sum += probs(1,0);
								   break;
							   }
						case 1:
							   if(WITH_NULL_TRIGGER){
								   prob_a_j /= min(I,WINDOW);//normalization
								   for(unsigned int k = 2;k <= min(I+1,WINDOW+1);++k){//trg.size()==I+2
									   WordVector wv = {trg[k-1],kNULL};
									   probs(2,0,k) = t2s.prob(wv,f_j) * prob_a_j;
									   sum += probs(2,0,k);
								   }
							   }
							   else{
								   WordVector wv = {kNULL,kNULL};
								   probs(2,0,0) = t2s.prob(wv,f_j) * prob_a_j;
								   sum += probs(2,0,0);
							   }
							   break;
						case 2:
							   if(WITH_NULL_TRIGGER){
								   prob_a_j /= min(I,WINDOW);//normalization
								   for(unsigned int k = 2;k <= min(I+1,WINDOW+1);++k){//trg.size()==I+2
									   WordVector wv = {kNULL,trg[k-1],kNULL};
									   probs(3,0,k,0) = t2s.prob(wv,f_j) * prob_a_j;
									   sum += probs(3,0,k,0);	
								   }
							   }
							   else{
								   WordVector wv = {kNULL,kNULL,kNULL};
								   probs(3,0,0,0) = t2s.prob(wv,f_j)*prob_a_j;
								   sum += probs(3,0,0,0);
							   }
							   break;
						default:
							   throw runtime_error("trigger is invalid");
							   break;
					}
				}
				double az = 0;
				if (favor_diagonal){
					az = DiagonalAlignment::ComputeZ(j + 1, src.size(), I, diagonal_tension) / prob_align_not_null;
				}
				for (unsigned i = 1; i <= I; ++i) {
					prob_a_j = 1.0 / (I + use_null); // uniform (model 1)
					if (favor_diagonal){
						prob_a_j = DiagonalAlignment::UnnormalizedProb(j + 1, i, src.size(), I, diagonal_tension) / az;
					}
					switch(TRIGGER){
						case 0:{
								   WordVector wv = {trg[i]};
								   probs(1,i+1) = t2s.prob(wv,f_j) * prob_a_j;
								   sum += probs(1,i+1);
								   break;
							   }
						case 1:
							   prob_a_j /= min(I - i + 1,WINDOW);
							   for(unsigned int k = i + 1; k < min(I+2,i+WINDOW+1); ++k){
								   WordVector wv = {trg[k],trg[i]};
								   probs(2,i+1,k+1) = t2s.prob(wv, f_j) * prob_a_j;
								   sum += probs(2,i+1,k+1);
							   }
							   break;
						case 2:
							   prob_a_j /= min((I - i + 1),WINDOW)*min(i,WINDOW);
							   for(unsigned int k = i + 1; k < min(I+2,i+WINDOW+1); ++k){
								   for(unsigned int l = 0;l < min(i,WINDOW);++l){
									   WordVector wv = {trg[l],trg[k],trg[i]};
									   probs(3,i+1,k+1,l+1) = t2s.prob(wv, f_j) * prob_a_j;
									   sum += probs(3,i+1,k+1,l+1);
								   }
							   }
							   break;
						default:
							   throw runtime_error("trigger is invalid");
							   break;
					}
				}
				// for(unsigned int ii = 1;ii <= I; ii++){
				// 	for(unsigned int jj = ii+1;jj<=I+1;++jj){
				// 		cout<<"probs("<<ii<<","<<jj<<")="<<probs(2,ii,jj)<<endl;
				// 	}
				// }
				if (final_iteration) {
					if (add_viterbi || write_alignments) {
						// WordID max_i = 0;
						double max_p = -1;
						int max_index = -1;
						if (use_null) {
							// max_i = kNULL;
							max_index = 0;
							max_p = 0;
							switch(TRIGGER){
								case 0:
									max_p = probs(1,0);
									break;
								case 1:
									if(WITH_NULL_TRIGGER){
										double tmpSum = 0.0; 
										for(unsigned int k = 2;k <= min(I + 1,WINDOW+1);++k){
											tmpSum += probs(2,0,k);
										}
										max_p = tmpSum;
									}
									else{
										max_p = probs(2,0,0);
									}
									break;
								case 2:
									if(WITH_NULL_TRIGGER){
										double tmpSum = 0.0;
										for(unsigned int k = 2;k <= min(I + 1,WINDOW+1);++k){
											tmpSum += probs(3,0,k,0);
										}
										max_p = tmpSum;
									}
									else{
										max_p = probs(3,0,0,0);	
									}
									break;
								default:
									throw runtime_error("trigger is invalid");
									break;
							}
						}
						for (unsigned int i = 1; i <= I; ++i) {
							switch(TRIGGER){
								case 0:
									if (probs(1,i+1) > max_p) {
										max_index = i;
										max_p = probs(1,i+1);
									}
									break;
								case 1:{
										   double tmpSum = 0.0;
										   for(unsigned int k = i + 1;k < min(I+2,i+WINDOW+1); ++k){
											   tmpSum += probs(2,i+1,k+1);
										   }
										   if(tmpSum > max_p){
											   max_index = i;
											   max_p = tmpSum;
										   }
										   break;
									   }
								case 2:{
										   double tmpSum = 0.0;
										   for(unsigned int k = i + 1;k < min(I+2,i+WINDOW+1);++k){
											   for(unsigned l = 0;l < min(i,WINDOW);++l){
												   tmpSum += probs(3,i+1,k+1,l+1);
											   }
										   }
										   if(tmpSum > max_p){
											   max_index = i;
											   max_p = tmpSum;
										   }
										   break;
									   }
								default:
									   throw runtime_error("trigger is invalid");
									   break;
							}
						}
						if (!hide_training_alignments && write_alignments) {
							if (max_index > 0) {
								if (first_al){
									first_al = false;
								} 
								else{
									fprintf(taof," ");
								}
								if (is_reverse){
									fprintf(taof,"%d-%d", max_index - 1, j);
								}
								else{
									fprintf(taof,"%d-%d", j , max_index - 1);
								}
							}
						}
					}
				} 
				else{
					if (use_null) {
						switch(TRIGGER){
							case 0:{
									   double count = probs(1,0) / sum;
									   WordVector wv = {kNULL};
									   c0 += count;
									   if(smooth == KN){
										   t2s.lm.addNgram(wv, f_j, count);
									   }
									   else{
										   t2s.Increment(wv, f_j, count);
									   }
									   break;
								   }
							case 1:
								   if(WITH_NULL_TRIGGER){
									   for(unsigned int k = 2; k <= min(I+1,WINDOW+1);++k){	
										   double count = probs(2,0,k)/sum;
										   c0 += count;
										   if(smooth == KN){
											   WordVector wv = {kNULL,trg[k]};
											   t2s.lm.addNgram(wv,f_j,count);
										   }
										   else{
											   WordVector wv = {trg[k],kNULL};
											   t2s.Increment(wv, f_j, count);
										   }
									   }
								   }	
								   else{
									   double count = probs(2,0,0)/sum;
									   WordVector wv = {kNULL,kNULL};
									   c0 += count;
									   if(smooth == KN){
										   t2s.lm.addNgram(wv,f_j,count);	
									   }
									   else{
										   t2s.Increment(wv, f_j, count);	
									   }
								   }
								   break;
							case 2:
								   if(WITH_NULL_TRIGGER){
									   for(unsigned int k = 2; k <= min(I+1,WINDOW+1); ++k){	
										   double count = probs(3,0,k,0)/sum;
										   WordVector wv = {kNULL,trg[k],kNULL};
										   c0 += count;
										   if(smooth == KN){
											   t2s.lm.addNgram(wv,f_j,count);	
										   }
										   else{
											   t2s.Increment(wv, f_j, count);	
										   }
									   }
								   }
								   else{
									   double count = probs(3,0,0,0)/sum;
									   WordVector wv = {kNULL,kNULL,kNULL};
									   c0 += count;	
									   if(smooth == KN){
										   t2s.lm.addNgram(wv,f_j,count);	
									   }
									   else{
										   t2s.Increment(wv, f_j, count);	
									   }
								   }		
								   break;
							default:
								   throw runtime_error("trigger is invalid");
								   break;
						}
					}
					switch(TRIGGER){
						case 0:
							for (unsigned int i = 1; i <= I; ++i) {
								const double p = probs(1,i+1) / sum;
								WordVector wv = {trg[i]};
								if(smooth == KN){ 
									t2s.lm.addNgram(wv,f_j,p);
								}
								else{
									t2s.Increment(wv, f_j, p);
								}
								emp_feat += DiagonalAlignment::Feature(j+1, i, src.size(), I) * p;//what is this line doing?
							}
							break;
						case 1:
							for (unsigned int i = 1; i <= I; ++i) {
								for(unsigned int k = i + 1; k < min(I+2,i+WINDOW+1); ++k){
									const double p = probs(2,i+1,k+1) / sum;
									if(smooth == KN){ 
										WordVector wv = {trg[i],trg[k]};//reversed vector
										t2s.lm.addNgram(wv,f_j,p);
									}
									else{
										WordVector wv = {trg[k],trg[i]};
										t2s.Increment(wv, f_j, p);

									}
									emp_feat += DiagonalAlignment::Feature(j+1, i,src.size(),I) * p;
								}
							}
							break;
						case 2:
							for(unsigned int i = 1;i <= I;++i){
								for(unsigned int k = i + 1; k < min(I+2,i+WINDOW+1);++k){
									for(unsigned int l = 0;l < min(i,WINDOW);++l){
										const double p = probs(3,i+1,k+1,l+1) / sum;
										if(smooth==KN){
											WordVector wv = {trg[i],trg[k],trg[l]};
											t2s.lm.addNgram(wv,f_j,p);
										}
										else{
											WordVector wv = {trg[l],trg[k],trg[i]};
											t2s.lm.addNgram(wv,f_j,p);
										}
										emp_feat += DiagonalAlignment::Feature(j+1, i, src.size(), I) * p;
									}
								}
							}
							break;
						default:
							throw runtime_error("trigger is invalid");
							break;
					}
				}
				likelihood += log(sum);
			}
			if (write_alignments && final_iteration && !hide_training_alignments){
				fprintf(taof,"\n");
			}
			// cout<<probs.toString()<<endl;
			probs.clear();
		}

		// log(e) = 1.0
		base2_likelihood = likelihood / log(2);

		if (flag) { 
			fprintf(tof,"\n");
			fprintf(stderr,"\n");
		}

		if (iter == 0) {
			mean_srclen_multiplier = tot_len_ratio / lc;
			cerr << "expected target length = source length * " << mean_srclen_multiplier << endl;
		}
		emp_feat /= toks;
		if(tof!=stderr){
			printProcess(tof);
		}
		printProcess(stderr);
		if (!final_iteration) {
			if (favor_diagonal && optimize_tension && iter > 0) {
				for (int ii = 0; ii < 8; ++ii) {
					double mod_feat = 0;
					unordered_map<pair<short,short>,unsigned,PairHash>::iterator it = size_counts.begin();
					for(; it != size_counts.end(); ++it) {
						const pair<short,short>& p = it->first;
						for (short j = 1; j <= p.first; ++j){
							mod_feat += it->second * DiagonalAlignment::ComputeDLogZ(j, p.first, p.second, diagonal_tension);
						}
					}
					mod_feat /= toks;
					fprintf(tof,"  %d model al-feat: %lf (tension=%lf)\n",ii+1,mod_feat,diagonal_tension);
					diagonal_tension += (emp_feat - mod_feat) * 20.0;
					if (diagonal_tension <= 0.1){
						diagonal_tension = 0.1;
					}
					if (diagonal_tension > 14){
						diagonal_tension = 14;
					}
				}
				fprintf(tof,"     final tension: %lf\n",diagonal_tension);
			}
			switch(smooth){
				case VB:
					t2s.NormalizeVB(alpha);
					break;
				case KN:
					if(semi_final_iteration){
						t2s.copyFromKneserNeyLM(true,true);
					}
					else{
						t2s.copyFromKneserNeyLM(false,false);
					}
					break;
				case NO:
				default:
					if(semi_final_iteration){
						t2s.Normalize(true);
					}
					else{
						t2s.Normalize(false);
					}
					break;
			}
			prob_align_not_null = 1.0 - prob_align_null;
		}
		if(ONE_INPUT_FILE){
			in.close();
		}
		else{
			fin.close();
			ein.close();	
		}
	}

	if(tof!=stderr){
		fclose(tof);
	}
	if(taof!=stdout){
		fclose(taof);
	}
	/************************endOfTraining*************************/

	/************************startOfTest***************************/
	if (DO_TEST) {
		double tlp = 0;
		int lc = 0;
		string line,fline,eline;
		ifstream in,fin,ein;
		tof= (conf.count("test_output_file")>0)?fopen(conf["test_output_file"].as<string>().c_str(),"w"):stderr;
		taof  = (conf.count("test_align_output_file")>0)?fopen(conf["test_align_output_file"].as<string>().c_str(),"w"):stdout;
		if(ONE_INPUT_FILE){
			in.open(testset.c_str(), ifstream::in);
			if (!in) {
				cerr << "Can't read " << testset << endl;
				return 1;
			}
		}
		else{
			fin.open(ftestset.c_str(), ifstream::in);
			ein.open(etestset.c_str(), ifstream::in);
			if(!fin){
				cerr << "Can't read " << ftestset << endl;
				return 1;
			}
			if(!ein){
				cerr << "Can't read " << etestset << endl;
				return 1;
			}
		}
		while (true) {
			++lc;
			vector<WordID> src, trg;
			if(ONE_TEST_FILE){
				getline(in,line);
			}
			else{
				getline(fin,fline);
				getline(ein,eline);
			}
			if (ONE_TEST_FILE && !in){
				break;
			}
			else if(!ONE_TEST_FILE && (!fin || !ein)){
				break;
			}

			if(ONE_TEST_FILE){
				ParseLine(line, &src, &trg,is_reverse);
			}
			else{
				ParseLineFromSeparateFiles(fline,eline,&src,&trg,is_reverse);
			}
			unsigned int I = trg.size()-2;//excluding <s> and </s>
			bool first_al = true;  // used for write_alignments
			double log_prob = Md::log_poisson(src.size(), 0.05 + I * mean_srclen_multiplier);//why 0.05 is needed
			// compute likelihood
			for (unsigned j = 0; j < src.size(); ++j) {
				const WordID& f_j = src[j];
				double sum = 0;
				int a_j = 0;
				double max_pat = 0;
				double prob_a_j = 1.0 / (I + use_null);  // uniform (model 1)
				if (use_null) {
					if (favor_diagonal){
						prob_a_j = prob_align_null;
					}
					switch(TRIGGER){
						case 0:{
								   WordVector wv = {kNULL};
								   max_pat = t2s.backoffProb(wv, f_j) * prob_a_j;
								   break;
							   }
						case 1:
							   if(WITH_NULL_TRIGGER){
								   double tmpSum = 0.0;
								   prob_a_j /= min(I,WINDOW);
								   for(unsigned int k = 1; k <= min(I+1,WINDOW);++k){	
									   WordVector wv = {trg[k],kNULL};
									   tmpSum += t2s.backoffProb(wv,f_j)*prob_a_j;
								   }
								   max_pat = max(max_pat,tmpSum);
							   }
							   else{
								   WordVector wv = {kNULL,kNULL};
								   max_pat = t2s.backoffProb(wv, f_j) * prob_a_j;
							   }
							   break;
						case 2:
							   if(WITH_NULL_TRIGGER){
								   double tmpSum = 0.0;
								   prob_a_j /= I;
								   for(unsigned int k = 1; k <= min(I+1,WINDOW);++k){	
									   WordVector wv = {kNULL,trg[k],kNULL};
									   tmpSum += t2s.backoffProb(wv,f_j)*prob_a_j;
								   }
								   max_pat = max(max_pat,tmpSum);
							   }
							   else{
								   WordVector wv = {kNULL,kNULL,kNULL};
								   max_pat = t2s.backoffProb(wv, f_j) * prob_a_j;
							   }	
							   break;
					}
					sum += max_pat;	
				}
				double az = 0;
				if (favor_diagonal){
					az = DiagonalAlignment::ComputeZ(j+1, src.size(), I, diagonal_tension) / prob_align_not_null;
				}
				for (unsigned int i = 1; i <= I; ++i) {
					prob_a_j = 1.0 / (I + use_null);  // uniform (model 1)
					if (favor_diagonal){
						prob_a_j = DiagonalAlignment::UnnormalizedProb(j + 1, i, src.size(), I , diagonal_tension) / az;
					}
					double pat = 0.0;
					switch(TRIGGER){
						case 0:{
								   WordVector wv = {trg[i]};
								   pat = t2s.backoffProb(wv, f_j) * prob_a_j;
								   if (pat > max_pat){
									   max_pat = pat;
									   a_j = i;
								   }
								   break;
							   }
						case 1:{
								   double tmpSum = 0.0;
								   prob_a_j /= min(I-i+1,WINDOW);
								   for(unsigned int k = i+1; k< min(I+2,i+WINDOW+1);++k){
									   WordVector wv = {trg[k],trg[i]};
									   tmpSum += t2s.backoffProb(wv,f_j)*prob_a_j;
								   }
								   if(tmpSum > max_pat){
									   max_pat = tmpSum;
									   a_j = i;
								   }
								   break;
							   }
						case 2:{
								   double tmpSum = 0.0;
								   prob_a_j /= min(I-i+1,WINDOW) * min(i,WINDOW);
								   for(unsigned int k = i+1; k< min(I+2,i+WINDOW+1);++k){
									   for(unsigned int l = 0;l < min(i,WINDOW);++l){
										   WordVector wv = {trg[l],trg[k],trg[i]};
										   tmpSum += t2s.backoffProb(wv,f_j)*prob_a_j;
									   }
								   }
								   if(tmpSum > max_pat){
									   max_pat = tmpSum;
									   a_j = i;
								   }
								   break;
							   }
						default:
							   throw runtime_error("trigger is invalid");
							   break;
					}
					sum += pat;
				}
				log_prob += log(sum);
				if (a_j > 0) {
					if (first_al){
						first_al = false;
					}
					else{
						fprintf(taof," ");
					}
					if (is_reverse){
						fprintf(taof, "%d-%d", a_j - 1, j);

					}
					else{	
						fprintf(taof,"%d-%d",j , a_j - 1);
					}
				}
			}
			tlp += log_prob;
			fprintf(taof,"\n");
		} // loop over test set sentences
		fprintf(tof,"TOTAL LOG PROB %lf\n",tlp);
		if(tof!=stderr){
			fclose(tof);
		}
		if(taof!=stdout){
			fclose(taof);
		}
	}
	if(show_ttable){
		t2s.ShowTTable(d);
	}
	return 0;
}

