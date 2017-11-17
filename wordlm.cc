#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/timing.h"
#include "dynet/rnn.h"
#include "dynet/gru.h"
#include "dynet/lstm.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/cfsm-builder.h"
#include "dynet/hsm-builder.h"
#include "dynet/globals.h"
#include "dynet/io.h"
#include "dynet/examples/cpp/utils/cl-args.h"

#include <iostream>
#include <fstream>
#include <regex>
#include <sstream>
#include <memory>

using namespace std;
using namespace dynet;

dynet::Dict d;
int kSOS;
int kEOS;

volatile bool INTERRUPTED = false;
bool SAMPLE = true;

unsigned LAYERS = 2;
unsigned INPUT_DIM = 40;
unsigned HIDDEN_DIM = 120;
unsigned VOCAB_SIZE = 0;

float DROPOUT = 0;
SoftmaxBuilder* cfsm = nullptr;

template <class Builder>
struct RNNLanguageModel {
  LookupParameter p_c;
  Parameter p_R;
  Parameter p_bias;
  Builder builder;

  explicit RNNLanguageModel(ParameterCollection& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model) {
    p_c = model.add_lookup_parameters(VOCAB_SIZE, {INPUT_DIM}); 
    p_R = model.add_parameters({VOCAB_SIZE, HIDDEN_DIM});
    p_bias = model.add_parameters({VOCAB_SIZE});
  }

  // return Expression of total loss
  Expression BuildLMGraph(const vector<int>& sent, ComputationGraph& cg, bool apply_dropout) {
    const unsigned slen = sent.size();
    if (apply_dropout) {
      builder.set_dropout(DROPOUT);
    } else {
      builder.disable_dropout();
    }
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
    Expression i_bias = parameter(cg, p_bias);  // word bias
    vector<Expression> errs;
    for (unsigned t = 0; t < slen; ++t) {
      Expression i_x_t = lookup(cg, p_c, sent[t]);
      Expression h_t = builder.add_input(i_x_t);
      Expression u_t = affine_transform({i_bias, i_R, h_t});   
      Expression i_err = pickneglogsoftmax(u_t, sent[t]);
      errs.push_back(i_err);
      }
      return sum(errs);
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
      Expression h_t = builder.add_input(i_x_t);
      Expression u_t = affine_transform({i_bias, i_R, h_t});
 
      Expression ydist = softmax(u_t);
      
      unsigned w = 0;
      while (w == 0 || (int)w == kSOS) {
        auto dist = as_vector(cg.incremental_forward(ydist));
        double p = rand01();
        for (; w < dist.size(); ++w) {
          p -= dist[w];
          if (p < 0.0) { break; }
        }
        if (w == dist.size()) w = kEOS;
      }
      cerr << (len == 1 ? "" : " ") << d.convert(w);
      cur = w;
    }
    cerr << endl;
  }
};

int main(int argc, char** argv) {
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);
  Params params; 
  if (argc != 3 && argc != 4) {
    cerr << "Usage: " << argv[0] << " corpus.txt dev.txt [model.params]\n";
    return 1;
  }
  kSOS = d.convert("始");
  kEOS = d.convert("終");
  if (params.dropout_rate)
    DROPOUT = params.dropout_rate;
  ParameterCollection model;
  float eta_decay_rate = params.eta_decay_rate;
  unsigned eta_decay_onset_epoch = params.eta_decay_onset_epoch; 
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
      training.push_back(read_sentence(line, d));
      ttoks += training.back().size();
    //put stuff back here later
    }
    cerr << tlc << " lines, " << ttoks << " tokens, " << d.size() << " types\n";
  }
  d.freeze(); // no new word types allowed
  VOCAB_SIZE = d.size();

  int dlc = 0;
  int dtoks = 0;
  cerr << "Reading dev data from " << argv[2] << "...\n";
  {
    ifstream in(argv[2]);
    assert(in);
    while(getline(in, line)) {
      ++dlc;
      dev.push_back(read_sentence(line, d));
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

  std::unique_ptr<Trainer> trainer(new SimpleSGDTrainer(model));
  //trainer->learning_rate = params.eta0;
  RNNLanguageModel<LSTMBuilder> lm(model);

  //RNNLanguageModel<SimpleRNNBuilder> lm(model);
  if (argc == 5) {
        string infname = argv[4];
        cerr << "Reading parameters from " << infname << "...\n";
        TextFileLoader loader(infname);
        loader.populate(model);
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
            if (first) { first = false; } else { trainer->update(); }
            //cerr << "**SHUFFLE\n";
            completed_epoch++;
            //if (eta_decay_onset_epoch && completed_epoch >= (int)eta_decay_onset_epoch) {trainer->learning_rate *= eta_decay_rate; }
            // Shuffle sentences
            shuffle(order.begin(), order.end(), *rndeng);
          }

          // Build graph for this instance
          ComputationGraph cg;
          auto& training_sentence = training[order[si]];
          chars += training_sentence.size();
          ++si;
          Expression loss_expr = lm.BuildLMGraph(training_sentence, cg, DROPOUT > 0.f);
          loss += as_scalar(cg.forward(loss_expr));
          cg.backward(loss_expr);
          trainer->update();
          ++lines;
      }
      report++;
      //sgd->status();
      cerr << '#' << report << " [epoch=" << (lines / training.size()) << "] E = " << (loss / chars) << " ppl=" << exp(loss / chars) << '\n';
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
          Expression loss_expr = lm.BuildLMGraph(dev_sentence, cg, DROPOUT > 0.f);
          dloss += as_scalar(cg.forward(loss_expr)); //check in the file thats having a problem, not able to find the index
          dchars += dev_sentence.size();
        }
        if (dloss < best) {
          best = dloss;
          TextFileSaver saver(fname);
          saver.save(model);
        }
        cerr << "\n***DEV [epoch=" << (lines / (double)training.size()) << "] E = " << (dloss / dchars) << " ppl=" << exp(dloss / dchars) << '\n';
      }
    }
  }
