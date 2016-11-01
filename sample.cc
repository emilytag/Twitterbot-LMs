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
#include "charlm.cc"

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

int main(int argc, char** argv) {
  cnn::Initialize(argc, argv);
  Model model;
  string infname = argv[1];
  cerr << "Reading parameters from " << infname << "...\n";
  ifstream in(infname);
  assert(in);
  boost::archive::text_iarchive ia(in);
  ia >> model;
  CharLSTM<LSTMBuilder> word_lstm(model);
  word_lstm.RandomSample();
}
