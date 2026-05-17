#ifndef UTIL_H
#define UTIL_H

#include <vector>

namespace Util {

template <typename T, typename U>
std::vector<T> subtract(const std::vector<T>& first, const std::vector<U>& second) {
    std::vector<T> output(first.size());
    for (size_t i = 0; i < first.size(); ++i) {
        output[i] = first[i] - second[i];
    }   
    return output;
}

} // namespace Util

#endif
