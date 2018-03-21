Not entirely sure how bevent works

open cmd and cd into this directory
type BEVENT.EXE bevent-h to see different commands

--

Expanded event descriptor, version 121(187) of 04/27/2016.
  Type 'bevent -h' for help.
Copyright 1989 Tom Tippett and David Nichols, 1993 David W. Smith.


bevent generates files suitable for use by dBase or Lotus-like programs.
Each record describes one event.
Usage: bevent [options] eventfile...
options:
  -h        print this help
  -i id     only process game given by id
  -q        ask whether to process each game
  -y year   Year to process (for teamyyyy and aaayyyy.ros).
  -s start  Earliest date to process (mmdd).
  -e end    Last date to process (mmdd).
  -a        generate Ascii-delimited format files (default)
  -ft       generate Fortran format files
  -m        use master player file instead of local roster files
  -f flist  give list of fields to output
              Default is 0-6,8-9,12-13,16-17,26-40,43-45,51,58-61
  -d        print list of field numbers and descriptions

--

right now it's telling me "Can't find teamfile(team)

-Ian