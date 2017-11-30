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
#include "dynet/examples/cpp-utils/cl-args.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>

using namespace std;
using namespace dynet;

unsigned LAYERS = 2; // Subject to change
unsigned INPUT_DIM = 50; // Embedding vector size
unsigned CHAR_DIM = 0; // Number of character types
unsigned HIDDEN_DIM = 300;  // Kind of arbitrary
float DROPOUT = 0;
bool SAMPLE = true;
SoftmaxBuilder* cfsm = nullptr;

dynet::Dict char_d;
int kSOS;
int kEOS;

volatile bool INTERRUPTED = false;

// Given the first character of a UTF8 block, find out how wide it is
// See http://en.wikipedia.org/wiki/UTF-8 for more info
inline unsigned int UTF8Len(unsigned char x) {
  if (x < 0x80) return 1;
  else if ((x >> 5) == 0x06) return 2;
  else if ((x >> 4) == 0x0e) return 3;
  else if ((x >> 3) == 0x1e) return 4;
  else if ((x >> 2) == 0x3e) return 5;
  else if ((x >> 1) == 0x7e) return 6;
  else abort();
}

inline vector<int> UTF8split(string s) {
  size_t cur = 0;
  vector<int> chars;
  while(cur < s.size()) {
    size_t len = UTF8Len(s[cur]);
    chars.push_back(char_d.convert(s.substr(cur, len)));
    cur += len;
  }

  return chars;
}

template <class Builder>
struct CharLSTM {
  LookupParameter p_char;
  Parameter p_R;
  Parameter p_bias;
  Builder builder;
  explicit CharLSTM(ParameterCollection& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, model) {
    p_char = model.add_lookup_parameters(CHAR_DIM, {INPUT_DIM}); 
    p_R = model.add_parameters({CHAR_DIM, HIDDEN_DIM});
    p_bias = model.add_parameters({CHAR_DIM});
  }

  // Builds the computation graphs and embeds a word
  Expression BuildGraph(const string& s, ComputationGraph& cg, bool apply_dropout) {
    vector<int> x = UTF8split(s);
    const unsigned slen = x.size();
    if (apply_dropout) {
      builder.set_dropout(DROPOUT);
    } else {
      builder.disable_dropout();
    }
    builder.new_graph(cg);
    builder.start_new_sequence();
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    vector<Expression> errs(slen + 1);
    Expression h_t = builder.add_input(lookup(cg, p_char, kSOS));
    // Adds each character to the LSTM
    for (unsigned i = 0; i < slen; ++i) {
      Expression u_t = affine_transform({i_bias, i_R, h_t});
      errs[i] = pickneglogsoftmax(u_t, x[i]);
      Expression x_t = lookup(cg, p_char, x[i]);
      h_t = builder.add_input(x_t);
    }
    Expression u_last = affine_transform({i_bias, i_R, h_t});
    errs.back() = pickneglogsoftmax(u_last, kEOS);
    return sum(errs);
  }

  // return Expression for total loss
  void RandomSample(int max_len = 140) {
    ComputationGraph cg;
    builder.new_graph(cg);  // reset RNN builder for new graph
    builder.start_new_sequence();
    
    Expression i_R = parameter(cg, p_R);
    Expression i_bias = parameter(cg, p_bias);
    Expression h_t = builder.add_input(lookup(cg, p_char, kSOS));
    int len = 0;
    int cur = kSOS;
    while(len < max_len && cur != kEOS) {
      Expression u_t = affine_transform({i_bias, i_R, h_t}); 
      Expression ydist = softmax(u_t);
      auto dist = as_vector(cg.incremental_forward(ydist));
      double p = rand01();
      cur = 0;
      for (; static_cast<unsigned>(cur) < dist.size(); ++cur) {
        p -= dist[cur];
        if (p < 0.0) { break; }
        }
        if (static_cast<unsigned>(cur) == dist.size()) cur = kEOS;
      if (cur == kEOS) break;
      ++len;
      cerr << char_d.convert(cur);
      Expression x_t = lookup(cg, p_char, cur);
      h_t = builder.add_input(x_t);
    }
    cerr << endl;
  }
};

class Sentence
{
  public:
    vector<string> words;
    vector<Expression> l2r_context; // Left to right context for each word
                                    // in sentence
};

vector<string> ReadData(string filename) {
  ifstream in(filename);
  {
    cerr << "Reading data from " << filename << " ...\n";
    string line;
    vector<string> sentences;
    while(getline(in, line)) {
      istringstream iss(line);
      sentences.push_back(line);
      // Count characters in the word form
      size_t cur = 0;
      while (cur < line.size()) {
          size_t len = UTF8Len(line[cur]);
          char_d.convert(line.substr(cur, len));
          cur += len;
      }
    }
    return sentences;
  }
};



int main(int argc, char** argv) {
  auto dyparams = dynet::extract_dynet_params(argc, argv);
  dynet::initialize(dyparams);
  Params params;
  bool isTrain = true;
  if (argc == 6){
    if (!strcmp(argv[5], "-t")) {
      isTrain = false;
    }
  }
  else if (argc != 4 && argc != 5 && argc != 6) {
    cerr << "Usage: " << argv[0] 
         << " train dev output.model [input.model] for training \n"
         << " train dev output.model [input.model] -t \n"; 
    return 1;
  }
  kSOS = char_d.convert("始");
  kEOS = char_d.convert("終");
  if (params.dropout_rate)
      DROPOUT = params.dropout_rate;
  ParameterCollection model;
  float eta_decay_rate = params.eta_decay_rate;
      unsigned eta_decay_onset_epoch = params.eta_decay_onset_epoch;
  vector<string> training = ReadData(argv[1]);
  vector<string> dev = ReadData(argv[2]);
  char_d.freeze(); //no new friends
  CHAR_DIM = char_d.size();
  std::unique_ptr<Trainer> trainer(new SimpleSGDTrainer(model));
  //trainer->learning_rate = params.eta0;
  CharLSTM<LSTMBuilder> lm(model);
  if (isTrain) {
      string fname = argv[3];
      if (argc==5) {
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
    double lines = 0;
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
            //if (eta_decay_onset_epoch && completed_epoch >= (int)eta_decay_onset_epoch)
              //trainer->learning_rate *= eta_decay_rate;
            // Shuffle sentences
            shuffle(order.begin(), order.end(), *rndeng);
          }
          // Build graph for this instance
          ComputationGraph cg;
          auto& training_sentence = training[order[si]];
          chars += training_sentence.size();
          ++si;
          Expression loss_expr = lm.BuildGraph(training_sentence, cg, DROPOUT > 0.f);
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
          Expression loss_expr = lm.BuildGraph(dev_sentence, cg, DROPOUT > 0.f);
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
  else {
 
   string infname = argv[4];
   cerr << "Reading parameters from " << infname << "...\n";
   TextFileLoader loader(infname);
   loader.populate(model);
   for (unsigned i = 0; i < 1000; ++i) lm.RandomSample();
  }
}

