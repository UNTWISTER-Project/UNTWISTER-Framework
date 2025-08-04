//
// Created by Kenneth Guldbrandt Lausdahl on 07/01/2020.
//

#ifndef RABBITMQFMUPROJECT_ISO8601TIMEPARSER_H
#define RABBITMQFMUPROJECT_ISO8601TIMEPARSER_H

#include <iostream>
#include <ctime>
#include "date/date.h"
#include <chrono>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace std::chrono;
using namespace date;

namespace Iso8601 {

    date::sys_time<std::chrono::milliseconds> parseIso8601ToMilliseconds(const std::string& input) ;
/*     std::string convertTimestampToString(const date::sys_time<std::chrono::milliseconds>& timestamp);
 */    std::string getCurrentTimestamp();
}

#endif //RABBITMQFMUPROJECT_ISO8601TIMEPARSER_H
