there are in total 14 rounds of data. Each round data contains n.xml (log file) and n.csv (ground truth special location)

n.csv formats as below:
 Time (relavent time that calculated by minus the first time value)+ lat1 + lng1 (these two values are directly from the App google map API) + x + y (these two are converted from lat&lng to utm) + lat +lng (these two are the relevent values which are conveinent for coordinate)