#!/usr/bin/env python3
import argparse
import math
from dataclasses import dataclass, field
from enum import Enum, auto
import os
import platform
import re
import sys
from typing import Union
import yaml


class TokenType(Enum):
    IDENTIFIER = auto()
    INTEGER = auto()
    COMMENT = auto()
    SYMBOL = auto()
    LABEL = auto()
    SENTINEL = auto()


@dataclass
class Token:
    value: Union[str, int]
    token_type: TokenType
    line_number: int


@dataclass
class InstructionSection:
    size: int
    offset: int
    section_type: list[str]
    values: dict[Union[str, int], int]
    default_value: Union[str, int]
    depends_section: int
    depends_attribute: str
    depends_mapping: Union[dict[str, int], list[int]]


@dataclass
class Instruction:
    sections: list[InstructionSection]
    num_bits_used: int = field(init=False)
    num_bits: int = field(init=False)

    def __post_init__(self):
        self.num_bits_used = max(map(lambda x: x.size + x.offset, self.sections), default=0)
        self.num_bits = int(8 * math.ceil(self.num_bits_used / 8))


architecture_name: Union[str, None] = None
language_file: Union[str, None] = None
level_name: Union[str, None] = None
program_name: Union[str, None] = None

identifier_pattern = re.compile(r'[a-zA-Z]\w+')
integer_pattern = re.compile(r'-?\d+')
comment_pattern = re.compile(r'#.+')
symbol_pattern = re.compile(r'\S+')
label_pattern = re.compile(r'label ([a-zA-Z]\w+):')

case_sensitive: bool = True
big_endian: bool = True

num_registers: int = -1
register_name_mapping: dict[str, int] = {}

instructions: list[Instruction] = []
instruction_names: set[str] = set()


def get_game_directory():
    my_os = platform.system()
    if my_os == 'Linux':
        return f'{os.getenv("HOME")}/.local/share/godot/app_userdata/Turing Complete'
    elif my_os == 'Windows':
        return f'{os.getenv("APPDATA")}\\godot\\app_userdata\\Turing Complete'
    elif my_os == 'Darwin':
        return f'{os.getenv("HOME")}/Library/Application Support/Godot/app_userdata/Turing Complete'


def require_key(dictionary: dict, key, msg: str):
    if key not in dictionary:
        raise KeyError(msg)


def main():
    global architecture_name
    global language_file
    global level_name
    global program_name

    parser = argparse.ArgumentParser(description='DynamicAssembler for Turing Complete')
    parser.add_argument('-a', '--architecture', help='name of your architecture')
    parser.add_argument('-c', '--config', help='config file for the parser')
    parser.add_argument('-l', '--language', help='language definition file')
    parser.add_argument('-m', '--map', help='map/level the program is for')
    parser.add_argument('-n', '--name', help='name of your program')
    parser.add_argument('file', help='file to be assembled by the assembler', nargs=1)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                parse_config(yaml.safe_load(f), os.path.dirname(args.config))
            except yaml.YAMLError as exc:
                print(exc)

    if args.architecture is not None:
        architecture_name = args.architecture
    if args.language is not None:
        language_file = args.language
    if args.map is not None:
        level_name = args.map
    if args.name is not None:
        program_name = args.name

    if language_file is None:
        print('You must specify a language in order to use this program')
        exit(1)

    with open(language_file, 'r') as f:
        try:
            parse_language(yaml.safe_load(f))
        except yaml.YAMLError as exc:
            print(exc)
    print(instruction_names)
    if args.file[0] == '-':
        data = assemble(sys.stdin.read())
    else:
        with open(args.file[0], 'r') as f:
            data = assemble(f.read())

    data_str = data_to_str(data)

    game_dir = get_game_directory()
    base_save_dir = os.path.join(game_dir, 'schematics/architecture')

    os.chdir(base_save_dir)
    if architecture_name is None:
        architectures = list(filter(os.path.isdir, os.listdir(os.curdir)))
        for arch in architectures:
            print(arch)
        architecture_name = input('Which architecture is this program for? ')
    os.chdir(architecture_name)
    if level_name is None:
        levels = list(filter(os.path.isdir, os.listdir(os.curdir)))
        for level in levels:
            print(level)
        level_name = input('Which level is this program for? ')
    os.chdir(level_name)
    if program_name is None:
        names = list(map(lambda x: x[:-9], filter(lambda x: x.endswith('.assembly'), os.listdir(os.curdir))))
        for name in names:
            print(name)
        program_name = input('What is the name of this program? ')
        pass

    save_file = os.path.join(base_save_dir, architecture_name, level_name, f'{program_name}.assembly')
    with open(save_file, 'w') as f:
        f.write(data_str)


def parse_config(config: dict, config_dir: str):
    global architecture_name
    global language_file
    global level_name
    global program_name

    if 'architecture' in config:
        architecture_name = config['architecture']
    if 'language' in config:
        language_file = os.path.join(config_dir, config['language'])
    if 'level' in config:
        level_name = config['level']
    if 'program_name' in config:
        program_name = config['program_name']


def parse_language(config: dict):
    global case_sensitive
    global big_endian
    global instructions
    global label_pattern

    printable_config = config.copy()
    del printable_config['registers']
    del printable_config['instructions']

    if 'case_sensitive' in config and not config['case_sensitive']:
        case_sensitive = False
        label_pattern = re.compile(label_pattern.pattern, re.IGNORECASE)

    if 'big_endian' in config and not config['big_endian']:
        big_endian = False

    require_key(config, 'registers', 'registers is not defined in the config file')
    parse_language_registers(config['registers'])

    require_key(config, 'instructions', 'instructions is not defined in the config file')
    parse_language_instructions(config['instructions'])


def parse_language_registers(config):
    global num_registers
    global register_name_mapping

    require_key(config, 'count', 'registers.count is not defined in the config file')
    num_registers = config['count']

    replace_default_names: bool = config['replace_default_names'] if 'replace_default_names' in config else False

    if not case_sensitive:
        for num in range(num_registers):
            if num in config:
                if not replace_default_names:
                    register_name_mapping[f'r{num}'.casefold()] = num
                for alias in config[num]:
                    register_name_mapping[alias.casefold()] = num
            else:
                register_name_mapping[f'r{num}'.casefold()] = num
    else:
        for num in range(num_registers):
            if num in config:
                if not replace_default_names:
                    register_name_mapping[f'r{num}'] = num
                for alias in config[num]:
                    register_name_mapping[alias] = num
            else:
                register_name_mapping[f'r{num}'] = num


def parse_language_instructions(config):
    global instructions
    global instruction_names

    for instr in config:
        sections: list[InstructionSection] = []
        for section in instr:
            require_key(section, 'bits', 'bits is not defined in an instruction')
            require_key(section, 'offset', 'offset is not defined in an instruction')
            require_key(section, 'type', 'type is not defined in an instruction')

            size: int = section['bits']
            offset: int = section['offset']
            section_type: Union[list[str], str] = section['type']
            if isinstance(section_type, str):
                section_type = [section_type]

            values: dict[Union[str, int], int] = {value: idx for idx, value in enumerate(section['values']) if value is not None} if 'values' in section else None
            default_value: Union[str, int] = section['default'] if 'default' in section else None

            if not case_sensitive:
                if values is not None:
                    for k in list(values.keys()):
                        if isinstance(k, str) and k != k.casefold():
                            values[k.casefold()] = values[k]
                            del values[k]
                if default_value is not None and isinstance(default_value, str):
                    default_value = default_value.casefold()

            if values is not None:
                instruction_names.update(values.keys())

            depends_section: Union[int, None] = None
            depends_attribute: Union[str, None] = None
            depends_mapping: Union[dict[str, int], list[int], None] = None
            if 'depends' in section:
                depends = section['depends']
                require_key(depends, 'section', 'section is not defined in a depends section of an instruction')
                require_key(depends, 'attribute', 'attribute is not defined in a depends section of an instruction')
                require_key(depends, 'mapping', 'mapping is not defined in a depends section of an instruction')

                depends_section = depends['section']
                depends_attribute = depends['attribute']
                depends_mapping = depends['mapping']

            sections.append(InstructionSection(size, offset, section_type, values, default_value, depends_section, depends_attribute, depends_mapping))

        instruction = Instruction(sections)

        # validate instruction

        bits_set = [0] * instruction.num_bits_used
        for section in instruction.sections:
            for i in range(section.size):
                bits_set[section.offset + i] += 1

        if any(map(lambda x: x - 1, bits_set)):
            raise Exception('an instruction sets at least one bit more than once or not at all')

        instructions.append(instruction)


def assemble(text: str) -> list[int]:
    global instructions

    tokens = tokenize(text)
    tokens.append(Token('', TokenType.SENTINEL, -1))

    label_map: dict[str, int] = {}
    decoded_instructions: list[tuple[int, list[tuple[str, int, int, Union[str, int, tuple[int, list[int]]]]]]] = []
    decoded_sections: list[tuple[str, int, int, Union[str, int, tuple[int, list[int]]]]] = []
    last_good_token_index: int = 0
    token_index: int = 0
    instr_index: int = 0
    section_index: int = 0
    while True:
        # reached the end of the token stream, but we still are assembling an instruction
        if token_index == len(tokens) - 1 and instr_index < len(instructions):
            instr = instructions[instr_index]
            section = instr.sections[section_index]
            # the rest of the instruction might not need input...
            if 'ignored' not in section.section_type and 'dependent' not in section.section_type:
                instr_index += 1
                section_index = 0
                token_index = last_good_token_index
                decoded_sections.clear()

        token = tokens[token_index]
        if token.token_type == TokenType.COMMENT:
            token_index += 1
            continue
        if token.token_type == TokenType.LABEL:
            offset: int = sum(map(lambda x: x[0], decoded_instructions))
            label_map[token.value] = offset
            token_index += 1
            continue

        # we failed to assemble a token from the defined instructions
        if instr_index == len(instructions):
            for i in decoded_instructions:
                print(i)
            raise Exception('failed to parse instruction')

        instr = instructions[instr_index]
        section = instr.sections[section_index]
        if len(set(section.section_type).intersection({'string', 'immediate', 'register', 'ignored', 'dependent'})) == 0:
            raise Exception(f'unknown instruction section type: {section.section_type}')

        parsed_section = False
        if 'string' in section.section_type:
            if token.token_type in [TokenType.SYMBOL, TokenType.IDENTIFIER] and token.value in section.values:
                decoded_sections.append(('string', section.offset, section.size, section.values[token.value]))
                section_index += 1
                token_index += 1
                parsed_section = True
            elif section.default_value is not None:
                decoded_sections.append(('string', section.offset, section.size, section.values[section.default_value]))
                section_index += 1
                parsed_section = True
        if not parsed_section and 'register' in section.section_type:
            if token.token_type == TokenType.IDENTIFIER and token.value in register_name_mapping:
                decoded_sections.append(('register', section.offset, section.size, register_name_mapping[token.value]))
                section_index += 1
                token_index += 1
                parsed_section = True
        if not parsed_section and 'immediate' in section.section_type:
            if token.token_type == TokenType.INTEGER or (token.token_type == TokenType.IDENTIFIER and token.value not in instruction_names):
                decoded_sections.append(('immediate', section.offset, section.size, token.value))
                section_index += 1
                token_index += 1
                parsed_section = True
        if not parsed_section and 'ignored' in section.section_type:
            decoded_sections.append(('ignored', section.offset, section.size, section.default_value))
            section_index += 1
            parsed_section = True
        if not parsed_section and 'dependent' in section.section_type:
            depended_section = decoded_sections[section.depends_section]
            depended_attribute = section.depends_attribute
            if depended_attribute == 'type':
                value = section.depends_mapping[depended_section[0]]
                decoded_sections.append(('dependent', section.offset, section.size, value))
                section_index += 1
            elif depended_attribute == 'value':
                if isinstance(depended_section[3], int):
                    value = section.depends_mapping[depended_section[3]]
                else:
                    # delegate mapping to convert_to_bytes since a label is preventing us from mapping here
                    value = (section.depends_section, section.depends_mapping)
                decoded_sections.append(('dependent', section.offset, section.size, value))
            else:
                raise Exception(f'invalid dependent attribute in instruction section: {depended_attribute}')
            parsed_section = True

        if not parsed_section:
            instr_index += 1
            section_index = 0
            token_index = last_good_token_index
            decoded_sections.clear()

        if section_index == len(instr.sections):
            section_index = 0
            instr_index = 0
            last_good_token_index = token_index
            decoded_instructions.append((instr.num_bits // 8, decoded_sections))
            decoded_sections = []
            if token_index == len(tokens) - 1:
                break

    byte_stream: list[int] = convert_to_bytes(decoded_instructions, label_map)
    return byte_stream


def convert_to_bytes(decoded_instructions: list[tuple[int, list[tuple[str, int, int, Union[str, int, tuple[int, list[int]]]]]]], label_map: dict[str, int]) -> list[int]:
    decoded_instrs = [x[1] for x in decoded_instructions]
    instruction_sizes = [x[0] for x in decoded_instructions]

    byte_stream: list[int] = []
    for idx, decoded_instr in enumerate(decoded_instrs):
        output_value: int = 0

        # finalize decoded instructions by mapping there sections to tuple[str, int, int, int]
        # labels are mapped to their offset and dependent instructions which are dependent on label resolution are resolved
        for i in range(len(decoded_instr)):
            section = decoded_instr[i]
            section_type, offset, size, value = section
            # map labels to ints
            if isinstance(value, str):
                decoded_instr[i] = (section_type, offset, size, label_map[value])
            # handle dependent types that were unable to map values because of labels
            elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], list) and all(map(lambda x: isinstance(x, int), value[1])):
                section_idx: int = value[0]
                mapping: list[int] = value[1]
                depended_section: tuple[str, int, int, int] = decoded_instr[section_idx]
                decoded_instr[i] = (section_type, offset, size, mapping[depended_section[3]])

        for section in decoded_instr:
            _, offset, size, value = section
            mask = (2 ** size - 1) << offset
            if (value & ~mask) >> (offset + size) not in [-1, 0]:
                raise Exception(f'invalid value for bit field of size {size}: {value}')

            output_value |= (value << offset) & mask

        instr_bytes: list[int] = []
        num_bytes = instruction_sizes[idx]
        for _ in range(num_bytes):
            instr_bytes.append(output_value & 0xFF)
            output_value >>= 8

        if big_endian:
            instr_bytes.reverse()
        byte_stream.extend(instr_bytes)
    return byte_stream


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        line = line.strip()
        while len(line) > 0:
            if match := label_pattern.match(line):
                if not case_sensitive:
                    tokens.append(Token(match.group(1).casefold(), TokenType.LABEL, idx))
                else:
                    tokens.append(Token(match.group(1), TokenType.LABEL, idx))
            elif match := identifier_pattern.match(line):
                if not case_sensitive:
                    tokens.append(Token(match.group().casefold(), TokenType.IDENTIFIER, idx))
                else:
                    tokens.append(Token(match.group(), TokenType.IDENTIFIER, idx))
            elif match := integer_pattern.match(line):
                tokens.append(Token(int(match.group()), TokenType.INTEGER, idx))
            elif match := comment_pattern.match(line):
                tokens.append(Token(match.group(), TokenType.COMMENT, idx))
            elif match := symbol_pattern.match(line):
                tokens.append(Token(match.group(), TokenType.SYMBOL, idx))
            else:
                raise Exception(f'Unable to tokenize line {idx} at \'{line}\'')
            line = line[match.span()[1]:].strip()

    return tokens


def data_to_str(data: list[int]) -> str:
    lines = []
    line = None
    for i in range(len(data)):
        if i & 7 == 0:
            if line is not None:
                lines.append(line)
            line = str(data[i])
        else:
            line += f' {data[i]}'

    if line is not None:
        lines.append(line)

    return '\n'.join(lines)


if __name__ == '__main__':
    main()
