#pragma once

#include <cstdint>
#include <vector>

#include "cutlass/numeric_types.h"

namespace host{
namespace utils {

///< store subbyte element in packed layout.
template <typename T>
class vector_subbyte {
 public:

  using Element = T;
  using Storage = uint8_t;

  static const int kSizeOfBit = cutlass::sizeof_bits<T>::value;

  /// Number of logical elements per stored object
  static const int kElementsPerStoredItem = int(sizeof(Storage) * 8) / kSizeOfBit;

  /// Bitmask for covering one item
  static Storage const kMask = ((Storage(1) << cutlass::sizeof_bits<T>::value) - 1);

  static_assert(kSizeOfBit < 8, "This vector must be used for subbytes.");

  typedef T value_type;
  typedef size_t size_type;
  typedef value_type *pointer;
  typedef value_type const *const_pointer;

  //
  // References
  //

  /// Reference object inserts or extracts sub-byte items
  class reference {
    /// Pointer to storage element
    Storage *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    /// Default ctor
    reference(): ptr_(nullptr), idx_(0) { }

    /// Ctor
    reference(Storage *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    /// Assignment
    reference &operator=(T x) {
      Storage item = (reinterpret_cast<Storage const &>(x) & kMask);

      Storage kUpdateMask = Storage(~(kMask << (idx_ * cutlass::sizeof_bits<T>::value)));
      *ptr_ = Storage(((*ptr_ & kUpdateMask) | (item << idx_ * cutlass::sizeof_bits<T>::value)));

      return *this;
    }

    T get() const {
      Storage item = Storage((*ptr_ >> (idx_ * cutlass::sizeof_bits<T>::value)) & kMask);
      return reinterpret_cast<T const &>(item);
    }

    /// Extract
    operator T() const {
      return get();
    }

    /// Explicit cast to int
    explicit operator int() const {
      return int(get());
    }

    /// Explicit cast to float
    explicit operator float() const {
      return float(get());
    }
  };

  /// Reference object extracts sub-byte items
  class const_reference {

    /// Pointer to storage element
    Storage const *ptr_;

    /// Index into elements packed into Storage object
    int idx_;

  public:

    /// Default ctor
    const_reference(): ptr_(nullptr), idx_(0) { }

    /// Ctor
    const_reference(Storage const *ptr, int idx = 0): ptr_(ptr), idx_(idx) { }

    const T get() const {
      Storage item = (*ptr_ >> (idx_ * cutlass::sizeof_bits<T>::value)) & kMask;
      return reinterpret_cast<T const &>(item);
    }

    /// Extract
    operator T() const {
      Storage item = Storage(Storage(*ptr_ >> Storage(idx_ * cutlass::sizeof_bits<T>::value)) & kMask);
      return reinterpret_cast<T const &>(item);
    }

    /// Explicit cast to int
    explicit operator int() const {
      return int(get());
    }

    /// Explicit cast to float
    explicit operator float() const {
      return float(get());
    }
  };

  vector_subbyte(size_type N, Element value) {
    num_elements = N;
    size_type storage_elements = calculate_capacity(num_elements);
    storage.resize(storage_elements);
    fill(value);
  }

 private:
  size_type num_elements = 0;
  std::vector<Storage> storage;

  size_type calculate_capacity(size_type N) {
    return (N + kElementsPerStoredItem - 1) / kElementsPerStoredItem;
  }

 public:
  void clear() {
    storage.clear();
  }

  void resize(size_type N) {
    num_elements = N;
    size_t storage_elements = calculate_capacity(num_elements);
    storage.resize(storage_elements);
  }

  reference at(size_type pos) {
    return reference(storage.data() + pos / kElementsPerStoredItem, pos % kElementsPerStoredItem);
  }

  const_reference at(size_type pos) const {
    return const_reference(storage.data() + pos / kElementsPerStoredItem, pos % kElementsPerStoredItem);
  }

  reference operator[](size_type pos) {
    return at(pos);
  }

  const_reference operator[](size_type pos) const {
    return at(pos);
  }

  pointer data() {
    return reinterpret_cast<pointer>(storage.data());
  }

  const_pointer data() const {
    return reinterpret_cast<const_pointer>(storage.data());
  }

  Storage * raw_data() {
    return storage.data();
  }

  Storage const * raw_data() const {
    return storage.data();
  }

  bool empty() const {
    return !num_elements;
  }

  size_type size() const {
    return num_elements;
  }

  size_type capacity() const {
    return storage.size();
  }

  void fill(T const &value) {

    for (int i = 0; i < kElementsPerStoredItem; ++i) {
      reference ref(storage.data(), i);
      ref = value;
    }

    for (size_t i = 1; i < storage.size(); ++i) {
      storage[i] = storage[0];
    }
  }

};
}  // utils
}  // namespace host
