/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#include "stdio.h"
#include "stdlib.h"
#include <locale.h>
#include <iostream>
#include "memory.h"
#include "fasttext.h"
#include "args.h"

using namespace fasttext;

void printUsage() {
  std::cout
    << "usage: fasttext <command> <args>\n\n"
    << "The commands supported by fasttext are:\n\n"
    << "  supervised          train a supervised classifier\n"
    << "  test                evaluate a supervised classifier\n"
    << "  predict             predict most likely labels\n"
    << "  predict-prob        predict most likely labels with probabilities\n"
    << "  skipgram            train a skipgram model\n"
    << "  cbow                train a cbow model\n"
    << "  print-vectors       print vectors given a trained model\n"
    << std::endl;
}

void printTestUsage() {
  std::cout
    << "usage: fasttext test <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPredictUsage() {
  std::cout
    << "usage: fasttext predict[-prob] <model> <test-data> [<k>]\n\n"
    << "  <model>      model filename\n"
    << "  <test-data>  test data filename (if -, read from stdin)\n"
    << "  <k>          (optional; 1 by default) predict top k labels\n"
    << std::endl;
}

void printPrintVectorsUsage() {
  std::cout
    << "usage: fasttext print-vectors <model>\n\n"
    << "  <model>      model filename\n"
    << std::endl;
}

void test(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printTestUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  std::string infile(argv[3]);
  if (infile == "-") {
    fasttext.test(std::cin, k);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Test file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.test(ifs, k);
    ifs.close();
  }
  exit(0);
}

void predict(int argc, char** argv) {
  int32_t k;
  if (argc == 4) {
    k = 1;
  } else if (argc == 5) {
    k = atoi(argv[4]);
  } else {
    printPredictUsage();
    exit(EXIT_FAILURE);
  }
  bool print_prob = std::string(argv[1]) == "predict-prob";
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));

  std::string infile(argv[3]);
  if (infile == "-") {
    fasttext.predict(std::cin, k, print_prob);
  } else {
    std::ifstream ifs(infile);
    if (!ifs.is_open()) {
      std::cerr << "Input file cannot be opened!" << std::endl;
      exit(EXIT_FAILURE);
    }
    fasttext.predict(ifs, k, print_prob);
    ifs.close();
  }

  exit(0);
}

void printVectors(int argc, char** argv) {
  if (argc != 3) {
    printPrintVectorsUsage();
    exit(EXIT_FAILURE);
  }
  FastText fasttext;
  fasttext.loadModel(std::string(argv[2]));
  fasttext.printVectors();
  exit(0);
}

void train(int argc, char** argv) {
  std::shared_ptr<Args> a = std::make_shared<Args>();
  a->parseArgs(argc, argv);
  FastText fasttext;
  fasttext.train(a);
}

extern "C" {
  std::vector<FastText*> fasttext_instance;

  void LoadModel(char* file_path, int idx = 0) {
    while (fasttext_instance.size() < idx + 1) {
      FastText* ft = new FastText();
      fasttext_instance.push_back(ft);
    }
    printf("file_path: %s\n", file_path);
    fasttext_instance[idx]->loadModel(std::string(file_path));
  }
  
  const char* Predict(char* input_text, int k, int idx = 0) {
    std::string input(input_text);
    const char* predict_ret = fasttext_instance[idx]->predict(input, k);
    return predict_ret;
  }

  bool IsCharacter(wchar_t w) {
    if ((w >= 'a' && w <= 'z') || (w >= 'A' && w <= 'Z')) {
      return true;
    }
    return false;
  }
  bool IsNumber(wchar_t w) {
    if (w >= '0' && w <= '9') {
      return true;
    }
    return false;
  }
  bool IsPunc(wchar_t w) {
    if (w < 128) {
      if (!IsCharacter(w) && !IsNumber(w)) {
        return true;
      }
    }
    return false;
  }

  const char* PreProcess(char* text, int length) {
    setlocale(LC_ALL, "zh_CN.utf8");
    //
    printf("Get the input: %s, %d\n", text, length);
    wchar_t * dBuf=NULL;
    int dSize = mbstowcs(dBuf, text, 0) + 1; 
    printf("need length: %d\n", dSize);
    dBuf=new wchar_t[dSize];
    wmemset(dBuf, 0, dSize);
    int nRet=mbstowcs(dBuf, text, length);
    //
    printf ("Wide char length: %d\n", nRet);

    if (nRet == -1) {
      return  NULL;
    } 
    
    std::vector<std::wstring> split_wstring;
    
    for (size_t i = 0; i < nRet; ++i) {
      if (dBuf[i] == ' ') {
        continue;
      }
      if (dBuf[i] >= 128) {
        split_wstring.push_back(std::wstring(dBuf + i, 1));
      } else if (IsCharacter(dBuf[i])) {
        if (i > 0 && (IsCharacter(dBuf[i]) || IsNumber(dBuf[i]) || dBuf[i] == '.')) {
          split_wstring[split_wstring.size() - 1].push_back(dBuf[i]);
        }
        split_wstring.push_back(std::wstring(dBuf + i, 1));
      } else if (IsNumber(dBuf[i])) {
        if (i > 0 && (IsCharacter(dBuf[i]) || IsNumber(dBuf[i]) || dBuf[i] == '.')) {
          split_wstring[split_wstring.size() - 1].push_back(dBuf[i]);
        }
        split_wstring.push_back(std::wstring(dBuf + i, 1));
      } else {
        split_wstring.push_back(std::wstring(dBuf + i, 1));
      }
    }

    std::wstring res_wstring;
    for (size_t i = 0; i < split_wstring.size(); ++i) {
      if (i != 0) {
          res_wstring.push_back(' ');
      }
      res_wstring.append(split_wstring[i]);
    }

    char* sBuf = NULL;
    dSize=wcstombs(sBuf, res_wstring.c_str(), 0)+1;
    sBuf = new char[dSize];
    memset(dBuf, 0, dSize);
    nRet=wcstombs(sBuf, res_wstring.c_str(), res_wstring.size());
    if (nRet == -1) {
      return NULL;
    }
  
    std::string output(sBuf);   
    delete [] sBuf;
    delete [] dBuf;
    return output.c_str();
  }
}



int main(int argc, char** argv) {
  if (argc < 2) {
    printUsage();
    exit(EXIT_FAILURE);
  }
  std::string command(argv[1]);
  if (command == "skipgram" || command == "cbow" || command == "supervised") {
    train(argc, argv);
  } else if (command == "test") {
    test(argc, argv);
  } else if (command == "print-vectors") {
    printVectors(argc, argv);
  } else if (command == "predict" || command == "predict-prob" ) {
    predict(argc, argv);
  } else {
    printUsage();
    exit(EXIT_FAILURE);
  }
  return 0;
}
