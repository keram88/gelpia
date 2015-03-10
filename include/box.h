
#ifndef BOX_H
#define BOX_H

#include "large_float.h" 
#include "interval.h"

typedef std::vector<interval_t> box_t;

class box {
  box_t value;
    
 public:
  box();
  box(const box &in);
  box(const box_t &in);

  int append(const std::string &low, const std::string &high);
  
  large_float width() const;
  int split_index() const;
  box midpoint() const;
  int size() const { return value.size(); }
  box_t get_value() const { return this->value; }

  box first(int index) const;
  box second(int index) const;
  
  ~box();
};


#endif
