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
#include <mutex>

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
  std::string str_buf;
  std::mutex mtx;

  void LoadModel(char* file_path, int idx = 0) {
    while (fasttext_instance.size() < idx + 1) {
      fasttext_instance.push_back(NULL);
    }
    if (fasttext_instance[idx] == NULL) {
      printf("load model from c code, file_path: %s\n", file_path);
      FastText* ft = new FastText();
      fasttext_instance[idx] = ft;
      fasttext_instance[idx]->loadModel(std::string(file_path));
    }
  }
  
  const char* Predict(char* input_text, int k, int idx = 0) {
    std::string input(input_text);
    const char* predict_ret = fasttext_instance[idx]->predict(input, k);
    return predict_ret;
  }
  const char* PreProcess(char* text);
  const char* PredictWithPreprocess(char* input_text, char* output, int k, int idx = 0) {
    const char* predict_ret = NULL;
    try
    {  
        std::lock_guard<std::mutex> lck(mtx);
        const char* temp = PreProcess(input_text);
        if (temp == NULL) {
          return NULL;
        } 
        std::string input(temp);
        predict_ret = fasttext_instance[idx]->predict(input, k);
        strcpy(output, predict_ret);
  
    }  
    catch (std::logic_error&e)  
    {  
        std::cout << e.what() << std::endl;  
        std::cout << "[exception caught]\n";  
    }  
   return predict_ret;
  }

  bool IsCharacter(wchar_t w) {
    if ((w >= L'a' && w <= L'z') || (w >= L'A' && w <= L'Z')) {
      return true;
    }
    return false;
  }
  bool IsNumber(wchar_t w) {
    if (w >= L'0' && w <= L'9') {
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

  const char* PreProcess(char* text) {
    setlocale(LC_ALL, "zh_CN.utf8");
    //
    int length = strlen(text);
    //printf("Get the input: %d %s\n", length, text);
    wchar_t * dBuf=NULL;
    int dSize = mbstowcs(dBuf, text, 0) + 1; 
    //printf("need length: %d\n", dSize);
    dBuf=new wchar_t[dSize];
    wmemset(dBuf, 0, dSize);
    int nRet=mbstowcs(dBuf, text, length);
    //
    //printf ("Wide char length: %d\n", nRet);

    if (nRet == -1) {
      return  NULL;
    } 
    
    std::vector<std::wstring> split_wstring;
    
    for (size_t i = 0; i < nRet; ++i) {
      if (dBuf[i] == L' ') {
        continue;
      }
      if (dBuf[i] >= 128) {
        split_wstring.push_back(std::wstring(dBuf + i, 1));
      } else if (IsCharacter(dBuf[i])) {
        if (i > 0 && (IsCharacter(dBuf[i - 1]) || IsNumber(dBuf[i - 1]) || dBuf[i - 1] == L'.')) {
          split_wstring[split_wstring.size() - 1].push_back(dBuf[i]);
        } else {
          split_wstring.push_back(std::wstring(dBuf + i, 1));
        }
      } else if (IsNumber(dBuf[i])) {
        if (i > 0 && (IsCharacter(dBuf[i - 1]) || IsNumber(dBuf[i - 1]) || dBuf[i - 1] == L'.')) {
          split_wstring[split_wstring.size() - 1].push_back(dBuf[i]);
        } else {
          split_wstring.push_back(std::wstring(dBuf + i, 1));
        }
      } else {
        split_wstring.push_back(std::wstring(dBuf + i, 1));
      }
    }
    //printf ("Split Wstring size: %d\n", split_wstring.size());

    std::wstring res_wstring;
    for (size_t i = 0; i < split_wstring.size(); ++i) {
      if (i != 0) {
          res_wstring.push_back(' ');
      }
      res_wstring.append(split_wstring[i]);
    }
    //printf ("res_wstring size: %d\n", res_wstring.size());


    char* sBuf = NULL;
    //dSize=wcstombs(sBuf, res_wstring.c_str(), 0)+1;
    //printf("Res char buffer size: %d\n", dSize);
    size_t char_buffer_size = res_wstring.size() * 4;
    sBuf = new char[char_buffer_size];
    memset(sBuf, 0, char_buffer_size);
    nRet=wcstombs(sBuf, res_wstring.c_str(), char_buffer_size);
    if (nRet == -1) {
      return NULL;
    }
    //printf ("oUTPUT LENGHT: %d %s\n", nRet, sBuf);
  
    str_buf.assign(sBuf);   
    delete [] sBuf;
    delete [] dBuf;
    return str_buf.c_str();
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
