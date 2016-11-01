#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/timing.h"
#include "cnn/rnn.h"
#include "cnn/gru.h"
#include "cnn/lstm.h"
#include "cnn/dict.h"
#include "cnn/expr.h"
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

cnn::Dict char_d;
int kSOS;
int kEOS;

volatile bool INTERRUPTED = false;
bool SAMPLE = true;

unsigned LAYERS = 2; // Subject to change
unsigned INPUT_DIM = 50; // Embedding vector size
unsigned CHAR_DIM = 0; // Number of character types
unsigned HIDDEN_DIM = 500;  // Kind of arbitrary

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
    chars.push_back(char_d.Convert(s.substr(cur, len)));
    cur += len;
  }

  return chars;
}

template <class Builder>
struct CharLSTM {
  LookupParameters* p_char;
  Parameters* p_R;
  Parameters* p_bias;
  Builder builder;
  explicit CharLSTM(Model& model) : builder(LAYERS, INPUT_DIM, HIDDEN_DIM, &model) {
    p_char = model.add_lookup_parameters(CHAR_DIM, {INPUT_DIM}); 
    p_R = model.add_parameters({CHAR_DIM, HIDDEN_DIM});
    p_bias = model.add_parameters({CHAR_DIM});
  }

  // Builds the computation graphs and embeds a word
  Expression BuildGraph(const string& s, ComputationGraph& cg) {
    vector<int> x = UTF8split(s);
    const unsigned slen = x.size();
    builder.new_graph(cg);
    builder.start_new_sequence();
    Expression R = parameter(cg, p_R);
    //vector<float> ok = as_vector(cg.incremental_forward());
    //for (std::vector<float>::const_iterator i = ok.begin(); i != ok.end(); ++i)
    //    std::cerr << *i << ' ';
    //cerr << "\n";
    Expression bias = parameter(cg, p_bias);
    vector<Expression> errs(slen + 1);

    Expression h_t = builder.add_input(lookup(cg, p_char, kSOS));
    // Adds each character to the LSTM
    for (unsigned i = 0; i < slen; ++i) {
      Expression u_t = affine_transform({bias, R, h_t});
      errs[i] = pickneglogsoftmax(u_t, x[i]);
      Expression x_t = lookup(cg, p_char, x[i]);
      h_t = builder.add_input(x_t);
    }
    Expression u_last = affine_transform({bias, R, h_t});
    errs.back() = pickneglogsoftmax(u_last, kEOS);
    Expression i_nerr = sum(errs);
    return i_nerr;
  }

  // return Expression for total loss
  void RandomSample(int max_len = 140) {
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
      Expression i_x_t = lookup(cg, p_char, cur);
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
      cerr << char_d.Convert(w);
      cur = w;
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
          char_d.Convert(line.substr(cur, len));
          cur += len;
      }
    }
    return sentences;
  }
};



int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);

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
  kSOS = char_d.Convert("始");
  kEOS = char_d.Convert("終");
  if (isTrain) {

      Model model;     

      vector<string> training = ReadData(argv[1]);
      vector<string> dev = ReadData(argv[2]);

      char_d.Freeze(); //no new friends
      char_d.SetUnk("<unk>");
      CHAR_DIM = char_d.size();

      Trainer* sgd = new SimpleSGDTrainer(&model);
      CharLSTM<LSTMBuilder> word_lstm(model);

      string fname = argv[3];

      if (argc==5) {
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
    double lines = 0;
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
          word_lstm.BuildGraph(training_sentence, cg);
          loss += as_scalar(cg.incremental_forward());
          cg.backward();
          sgd->update();
          ++lines;
      }
      report++;
      //sgd->status();
      cerr << '#' << report << " [epoch=" << (lines / training.size()) << " eta=" << sgd->eta << "] E = " << (loss / chars) << " ppl=" << exp(loss / chars) << '\n';
      if (report % dev_every_i_reports == 0) {
        word_lstm.RandomSample();
        word_lstm.RandomSample();
        word_lstm.RandomSample();
        word_lstm.RandomSample();
        word_lstm.RandomSample();
        double dloss = 0;
        int dchars = 0;
        for (auto& dev_sentence : dev) {
          ComputationGraph cg;
          word_lstm.BuildGraph(dev_sentence, cg);
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
}

