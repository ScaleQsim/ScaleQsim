// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IO_FILE_H_
#define IO_FILE_H_

#include <cstdint>
#include <fstream>
#include <string>

#include "io.h"

namespace qsim {

/**
 * Controller for output logs with methods for writing to file.
 */
struct IOFile : public IO {
  static std::ifstream StreamFromFile(const std::string& file) {
    std::ifstream fs;
    fs.open(file);
    if (!fs) {
      errorf("cannot open %s for reading.\n", file.c_str());
    }
    return fs;
  }

  static void CloseStream(std::ifstream& fs) {
    fs.close();
  }

  static bool WriteToFile(
      const std::string& file, const std::string& content) {
    return WriteToFile(file, content.data(), content.size());
  }

  /*  
  static bool WriteToFile(
      const std::string& file, const void* data, uint64_t size) {
    auto fs = std::fstream(file, std::ios::out | std::ios::binary);

    if (!fs) {
      errorf("cannot open %s for writing.\n", file.c_str());
      return false;
    } else {
      fs.write((const char*) data, size);
      if (!fs) {
        errorf("cannot write to %s.\n", file.c_str());
        return false;
      }

      fs.close();
    }

    return true;
  }
  */

static bool WriteToFile(const std::string& file, const void* data, uint64_t size) {
    IO::messagef("Attempting to write to file %s\n", file.c_str()); // 디버깅 출력
    auto fs = std::fstream(file, std::ios::out | std::ios::binary);
    if (!fs) {
        IO::errorf("Cannot open %s for writing.\n", file.c_str());
        return false;
    }
    fs.write((const char*) data, size);
    if (!fs) {
        IO::errorf("Cannot write to %s.\n", file.c_str());
        return false;
    }
    fs.close();
    IO::messagef("Write to file %s successful.\n", file.c_str());
    return true;
}

};

}  // namespace qsim

#endif  // IO_FILE_H_
