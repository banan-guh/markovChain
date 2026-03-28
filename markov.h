#ifndef MARKOV_H
#define MARKOV_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

class Markov {
private:
  int pick_weighted(std::map<int, int>& options, bool f);
  int pick_random(std::map<int, int>& options, bool f);

public:
  int START = 0;
  int END = 1;

  std::vector<std::string> vocabulary;
  std::unordered_map<std::string, int> word_to_id;
  std::map<std::vector<int>, std::map<int, int>> memory;
  std::map<std::vector<int>, std::map<int, int>> reverse_memory;

  Markov();
  int get_id(std::string word);
  std::string sanitize(std::string raw);
  std::string generate(int o, bool w, int c, bool r, bool f);
  std::string generate_seeded(std::string seed, int o, bool w, int c, bool r, bool f);
  void train(std::string raw_message, int max_order);
  void train_from_file(std::string filename, int o);
  void save_brain(std::string folder);
  void load_brain(std::string folder);
  void purge(std::vector<std::string> blocked_words);
};

#endif