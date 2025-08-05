/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_CORE_PRINT_H
#define OV_CORE_PRINT_H

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

namespace ov_core {

/**
 * @brief open_vins 的打印器，允许进行不同级别的打印输出
 *
 * 设置全局详细程度的方法如下：
 * @code{.cpp}
 * ov_core::Printer::setPrintLevel("WARNING");
 * ov_core::Printer::setPrintLevel(ov_core::Printer::PrintLevel::WARNING);
 * @endcode
 */
class Printer {
public:
  /**
   * @brief 可用的不同打印级别
   *
   * - PrintLevel::ALL : 所有 PRINT_XXXX 都会输出到控制台
   * - PrintLevel::DEBUG : 会打印 "DEBUG"、"INFO"、"WARNING" 和 "ERROR"。 "ALL" 会被屏蔽
   * - PrintLevel::INFO : 会打印 "INFO"、"WARNING" 和 "ERROR"。 "ALL" 和 "DEBUG" 会被屏蔽
   * - PrintLevel::WARNING : 只会打印 "WARNING" 和 "ERROR"。 "ALL"、"DEBUG" 和 "INFO" 会被屏蔽
   * - PrintLevel::ERROR : 只会打印 "ERROR"。其他全部屏蔽
   * - PrintLevel::SILENT : 所有 PRINT_XXXX 都会被屏蔽
   */
  enum PrintLevel { ALL = 0, DEBUG = 1, INFO = 2, WARNING = 3, ERROR = 4, SILENT = 5 };

  /**
   * @brief 设置所有后续标准输出的打印级别
   * @param level 要使用的调试级别
   */
  static void setPrintLevel(const std::string &level);

  /**
   * @brief 设置所有后续标准输出的打印级别
   * @param level 要使用的调试级别
   */
  static void setPrintLevel(PrintLevel level);

  /**
   * @brief 打印到标准输出的函数
   * @param level 此次打印调用的级别
   * @param location 打印发生的位置
   * @param line 打印发生的行号
   * @param format printf 格式
   */
  static void debugPrint(PrintLevel level, const char location[], const char line[], const char *format, ...);

  /// 当前的打印级别
  static PrintLevel current_print_level;

private:
  /// 文件路径的最大长度。用于避免非常长的文件路径
  static constexpr uint32_t MAX_FILE_PATH_LEGTH = 30;
};

} /* namespace ov_core */

/*
 * 将任意内容转换为字符串
 */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/*
 * 不同类型的打印级别
 */
#define PRINT_ALL(x...) ov_core::Printer::debugPrint(ov_core::Printer::PrintLevel::ALL, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_DEBUG(x...) ov_core::Printer::debugPrint(ov_core::Printer::PrintLevel::DEBUG, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_INFO(x...) ov_core::Printer::debugPrint(ov_core::Printer::PrintLevel::INFO, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_WARNING(x...) ov_core::Printer::debugPrint(ov_core::Printer::PrintLevel::WARNING, __FILE__, TOSTRING(__LINE__), x);
#define PRINT_ERROR(x...) ov_core::Printer::debugPrint(ov_core::Printer::PrintLevel::ERROR, __FILE__, TOSTRING(__LINE__), x);

#endif /* OV_CORE_PRINT_H */
