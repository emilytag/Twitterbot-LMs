#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
# include "cnn/expr.h"
#include "cnn/cfsm-builder.h"
#include "cnn/hsm-builder.h"

#include <algorithm>
#include <iostream>
#include <fstream>
#include <regex>
#include <sstream>
#include <cstdlib>

#include <boost/algorithm/string/join.hpp>
#include <boost/program_options.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/regex.hpp>

using namespace std;
using namespace cnn;

cnn::Dict d;
int kSOS;
int kEOS;

volatile bool INTERRUPTED = false;
bool SAMPLE = true;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 120;
unsigned VOCAB_SIZE = 0;


template <class Builder>
struct RNNLanguageModel {
  LookupParameters* p_c;
  Parameters* p_R;
  Parameters* p_bias;
  Builder builder;

  explicit RNNLanguageModel(Model& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
  }

  // return Expression of total loss
  Expression BuildLMGraph(const vector<int>& sent, ComputationGraph& cg) {
    const unsigned slen = sent.size() - 1;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    vector<Expression> errs;
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_r_t =  i_bias + i_R * i_y_t;
      
      // LogSoftmax followed by PickElement can be written in one step
      // using PickNegLogSoftmax
#if 0
      Expression i_ydist = logsoftmax(i_r_t);
      errs.push_back(pick(i_ydist, sent[t+1]));
#if 0
      Expression i_ydist = softmax(i_r_t);
      i_ydist = log(i_ydist)
      errs.push_back(pick(i_ydist, sent[t+1]));
#endif
#else
      Expression i_err = pickneglogsoftmax(i_r_t, sent[t+1]);
      errs.push_back(i_err);
#endif
    }
    Expression i_nerr = sum(errs);
#if 0
    return -i_nerr;
#else
    return i_nerr;
#endif
  }

  // return Expression for total loss
  void RandomSample(int max_len = 150) {
    cerr << endl;
    ComputationGraph cg;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    vector<Expression> errs;
    int len = 0;
    int cur = kSOS;
    while(len < max_len && cur != kEOS) {
      ++len;
      Expression i_x_t = lookup(cg, p_c, cur);
      // y_t = RNN(x_t)
      Expression i_y_t = builder.add_input(i_x_t);
      Expression i_r_t = i_bias + i_R * i_y_t;
      
      Expression ydist = softmax(i_r_t);
      
      unsigned w = 0;
      while (w == 0 || (int)w == kSOS) {
        auto dist = as_vector(cg.incremental_forward());
        double p = rand01();
        for (; w < dist.size(); ++w) {
          p -= dist[w];
          if (p < 0.0) { break; }
        }
        if (w == dist.size()) w = kEOS;
      }
      cerr << (len == 1 ? "" : " ") << d.Convert(w);
      cur = w;
    }
    cerr << endl;
  }
};

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kSOS = d.Convert("始");
  kEOS = d.Convert("終");
  Model model;  
  vector<vector<int>> training, dev;
  string line;
  int tlc = 0;
  int ttoks = 0;
  cerr << "Reading training data from " << argv[1] << "...\n";
  {
    ifstream in(argv[1]);
    assert(in);
    while(getline(in, line)) {
      ++tlc;
      training.push_back(ReadSentence(line, &d));
      ttoks += training.back().size();
    //put stuff back here later
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.Freeze(); // no new word types allowed
  d.SetUnk("UNK");
  VOCAB_SIZE = d.size();

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      dev.push_back(ReadSentence(line, &d));
      dtoks += dev.back().size();
    }
    cerr << dlc << " lines, " << dtoks << " tokens\n";
  }
  ostringstream os;
  os << "lm"
     << '_' << LAYERS
     << '_' << INPUT_DIM
     << '_' << HIDDEN_DIM
     << "-pid" << getpid() << ".params";
  const string fname = os.str();
  cerr << "Parameters will be written to: " << fname << endl;

  bool use_momentum = false;
  //if (use_momentum)
  //  sgd = new MomentumSGDTrainer(&model);
  //else
  Trainer* sgd = new SimpleSGDTrainer(&model);
  RNNLanguageModel<LSTMBuilder> lm(model);

  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (argc == 5) {
        string infname = argv[4];
        cerr << "Reading parameters from " << infname << "...\n";
        ifstream in(infname);
        assert(in);
        boost::archive::text_iarchive ia(in);
        ia >> model;
  }

  double best = 9e+99;
  unsigned report_every_i = 100;
  unsigned dev_every_i_reports = 10;
  unsigned si = training.size();
  if (report_every_i > si) report_every_i = si;
  vector<unsigned> order(training.size());
  for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
  bool first = true;
  int report = 0;
  unsigned lines = 0;
    int completed_epoch = -1;
    while(!INTERRUPTED) {
      double loss = 0;
      unsigned chars = 0;
      for (unsigned i = 0; i < report_every_i; ++i) {
          if (si == training.size()) {
            si = 0;
            if (first) { first = false; } else { sgd->update_epoch(); }
            //cerr << "**SHUFFLE\n";
            completed_epoch++;
            // Shuffle sentences
            shuffle(order.begin(), order.end(), *rndeng);
          }

          // Build graph for this instance
          ComputationGraph cg;
          auto& training_sentence = training[order[si]];
          chars += training_sentence.size();
          ++si;
          lm.BuildLMGraph(training_sentence, cg);
          loss += as_scalar(cg.incremental_forward());
          cg.backward();
          sgd->update();
          ++lines;
      }
      report++;
      //sgd->status();
      cerr << '#' << report << " [epoch=" << (lines / training.size()) << " eta=" << sgd->eta << "] E = " << (loss / chars) << " ppl=" << exp(loss / chars) << '\n';
      if (report % dev_every_i_reports == 0) {
        lm.RandomSample();
        lm.RandomSample();
        lm.RandomSample();
        lm.RandomSample();
        lm.RandomSample();
        double dloss = 0;
        int dchars = 0;
        for (auto& dev_sentence : dev) {
          ComputationGraph cg;
          lm.BuildLMGraph(dev_sentence, cg);
          dloss += as_scalar(cg.forward()); //check in the file thats having a problem, not able to find the index
          dchars += dev_sentence.size();
        }
        if (dloss < best) {
          best = dloss;
          ofstream out(fname);
          boost::archive::text_oarchive oa(out);
          oa << model;
        }
        cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << '\n';
      }
    }
    delete sgd;
  }
