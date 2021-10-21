# [Turing Complete](https://turingcomplete.game/) Dynamic Assembler

---

This program is designed as a more performant assembler for the game Turing Complete than the built in assembler.
It works by having you define each instruction group and how it maps to a binary output. This is then used to parse
a text file you provide the program and it then will write the compiled data into the game's directory so that it
can be used by the computer you designed within the game.

---

## NOTE: There are likely still bugs in here and the error messages are very unhelpful

I plan to fix this soon but, as long as you define your language file correctly and don't have any typos in your
program, it should work fine.

---

## How to run this program

1. Install python 3.9+ (3.6 and above might work fine, but I wrote this with 3.9 installed on my system)
2. Clone this git repo
3. Run `python -m pip install -r requirements.txt` from a terminal window
4. If you're on linux or mac, you should be able to directly run the program with `./main.py [arguments]`
   1. Otherwise you can run it with `python main.py [arguments]`

---

## Program Arguments

- You are required to either pass in a config file that lists the language specification inside or directly specify
the language specification from the command line.
- You are required to use the assembly program file as the last argument
- The architecture, level, and program name CAN be defined in the config or passed in directly, but they are not
required since the program will ask for them once it's compiled the assembly program

```
usage: main.py [-h] [-a ARCHITECTURE] [-c CONFIG] [-l LANGUAGE] [-m MAP] [-n NAME] file

DynamicAssembler for Turing Complete

positional arguments:
  file                  file to be assembled by the assembler

optional arguments:
  -h, --help            show this help message and exit
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        name of your architecture
  -c CONFIG, --config CONFIG
                        config file for the parser
  -l LANGUAGE, --language LANGUAGE
                        language definition file
  -m MAP, --map MAP     map/level the program is for
  -n NAME, --name NAME  name of your program
```