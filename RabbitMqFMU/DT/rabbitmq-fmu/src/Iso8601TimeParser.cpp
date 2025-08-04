//
// Created by Kenneth Guldbrandt Lausdahl on 07/01/2020.
//

#include "Iso8601TimeParser.h"

#include <iostream>

namespace Iso8601 {

    date::sys_time<std::chrono::milliseconds> parseIso8601ToMilliseconds(const std::string& input) {
        std::istringstream in{input};
        date::sys_time<std::chrono::milliseconds> tp;
        in >> date::parse("%FT%T%Z", tp);
        if (in.fail()) {
            cout << "fail on: "<<input<<endl;
            in.clear();
            in.exceptions(std::ios::failbit);
            in.str(input);
            in >> date::parse("%FT%T%Ez", tp);
            if (in.fail())
            {
                cout << "fail2 on: "<<input<<endl;
            }
        }
        return tp;
    }

    /* std::string convertTimestampToString(const date::sys_time<std::chrono::milliseconds>& timestamp) {
        // Format the timestamp
        std::stringstream formattedTimestamp;
        formattedTimestamp << date::format("%FT%T", timestamp);
        
        // Append "+00:00" as the UTC offset
        formattedTimestamp << "+00:00";

        return formattedTimestamp.str();
    } */

   std::string getCurrentTimestamp() {
        time_t now;
        time(&now);
        char buf[sizeof "2011-10-08T07:07:09Z"];
        strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));
        return buf;
    }
}