#!/usr/bin/env python3

import argparse
import ast
import hashlib
import os.path as path
import sys
import re

from time import time

import ian_utils as iu

from function_to_lexed import SYMBOLIC_CONSTS
from lexed_to_parsed import parse_function
from pass_lift_inputs_and_assigns import lift_inputs_and_assigns
from pass_lift_consts import lift_consts
from pass_single_assignment import single_assignment
from pass_reverse_diff import reverse_diff
from pass_simplify import simplify
from pass_dead_removal import dead_removal
from pass_utils import gaol_eval
from output_rust import to_rust
from output_interp import to_interp
from output_flatten import flatten

from input_parser import process
from gelpia import base_dir

def parse_args():
    exe = path.basename(sys.argv[0])
    arg_parser = create_common_option_parser(exe == "gelpia")

    if exe == "dop_gelpia":
        (args, function, epsilons) = add_dop_args(arg_parser)
    else:
        if exe != "gelpia":
            print("Defaulting to gelpia argument parsing")
        (args, function, epsilons) = add_gelpia_args(arg_parser)

    return finish_parsing_args(args, function, epsilons)





def create_common_option_parser(use_ampersand):
    arg_parser = iu.IanArgumentParser(description="Global function optimizer based "
                                      "on branch and bound for noncontinuous "
                                      "functions.",
                                      fromfile_prefix_chars= '@' if use_ampersand else None)
    arg_parser.add_argument("-v", "--verbose", help="increase output verbosity",
                            type=int, default=0)
    arg_parser.add_argument("-d", "--debug",
                        help="Debug run of function. Makes the minimum verbosity"
                        " level one. Runs a debug build of gelpia, "
                        "with backtrace enabled.", action="store_true")
    arg_parser.add_argument("--dreal", action='store_const', const=True, default=False)
    arg_parser.add_argument("-t", "--timeout",
                        type=int, help="Timeout for execution in seconds.",
                        default=0)
    arg_parser.add_argument("-s", "--seed",
                        type=int, help="Optional seed (u32) for the random number generators used within gelpia.\n"
                            "A value of 0 (default) indicates to use the default seed, a value of 1 indicates gelpia \n"
                            "will use a randomly selected seed. Any other value will be used as the RNG seed.",
                            default=0)
    arg_parser.add_argument("-M", "--maxiters",
                        type=int, help="Maximum IBBA iterations.",
                        default=0)
    arg_parser.add_argument("-g", "--grace",
                        type=int, help="Grace period for timeout option. Defaults to twice the supplied timeout",
                        default=0)
    arg_parser.add_argument("-u", "--update",
                        type=int, help="Time between update thread executions.",
                            default=0)
    arg_parser.add_argument("-L", "--logging",
                        help="Enable solver logging to stderr",
                        type=str, nargs='?', const=True, default=None)
    arg_parser.add_argument("-T", "--fptaylor",
                        help="FPTaylor compatibility",
                            type=str, nargs='?', const=True, default=False)
    arg_parser.add_argument("-z", "--skip-div-zero",
                            action="store_true", help="Skip division by zero check")
    arg_parser.add_argument("-ie", "--input-epsilon",
                        help="cuttoff for function input size",
                            type=float, default=None)
    arg_parser.add_argument("-oer", "--relative-input-epsilon",
                            help="relative error cutoff for function output size",
                            type=float, default=None);
    arg_parser.add_argument("-oe", "--output-epsilon",
                            help="cuttoff for function output size",
                            type=float, default=None)
    arg_parser.add_argument("-lq", "--log-query",
                            help="Saves a copy of the query for later examination/benchmarking",
                            action="store_true")
    arg_parser.add_argument("-c", "--concurrency", help = "Threads per process",
                            type=int, required=True,)
    arg_parser.add_argument("-n", "--np", help = "Number of procs",
                            type=int, required=True,)

    return arg_parser




def parse_input_box(box):
    reformatted = list()
    names = set()
    for i in box:
      name = i[0]
      if name in names:
        print("Duplicate variable", name)
        exit(-1)
      names.add(name)
      reformatted.append("{} = [{},{}];".format(name, *i[1]))
    return '\n'.join(reformatted)

def add_gelpia_args(arg_parser):
    """ Command line argument parser. Returns a dict from arg name to value"""

    arg_parser.add_argument("-i", "--input",
                        help="Search space. "
                        "Format is: {V1 : (inf_V1, sup_V1), ...}"
                        "Where V1 is the interval name, inf_V1 is the infimum, "
                        "and sup_V1 is the supremum",
                        type=str, nargs='+', required=True,)
    arg_parser.add_argument("-f", "--function",
                        help="the c++ interval arithmatic function to evaluate",
                        type=str, nargs='+', required=True,)



    # actually parse
    args = arg_parser.parse_args()

    # dump query for later examination/benchmarking
    if args.log_query:
        qlog_dir = path.join(base_dir, "query_log")
        base = "fptaylor_" if args.fptaylor else ""
        core = "query_"+hashlib.sha224(bytes(''.join(args.function)+''.join(args.input), "utf-8")).hexdigest()
        fname = base+core+"_"+str(1)
        i = 2
        while path.isfile(path.join(qlog_dir, fname+".txt")):
            fname = base+core+"_"+str(i)
            i += 1
        with open(path.join(qlog_dir, fname+".txt"), 'w') as f:
            if args.dreal:
                f.write("--dreal")
            f.write('--input "{}"\n'.format(args.input))
            f.write('--function "{}"\n\n'.format(args.function))


    # reformat query
    function = ' '.join(args.function)
    if args.dreal:
        function = "-({})".format(function)

    inputs = ' '.join(args.input)
    start = parse_input_box(process(inputs))

    reformatted_query = start+'\n'+function

    ie = oe = 0.001
    oer = 0
    if args.input_epsilon != None:
        ie = args.input_epsilon
    if args.output_epsilon != None:
        oe = args.output_epsilon
    if args.output_epsilon != None:
        oe = args.output_epsilon
    if args.relative_input_epsilon != None:
        oer = args.relative_input_epsilon
    return (args,
            reformatted_query,
            [ie, oe, oer])





def add_dop_args(arg_parser):
    arg_parser.add_argument("query_file",type=str)
    arg_parser.add_argument("-p", "--prec",
                        help="dOp delta precision",
                            type=float, default=None)
    args = arg_parser.parse_args()
    with open(args.query_file, 'r') as f:
        query = f.read()


    # precision
    pmatch = re.match(r"^prec: +(\d*.\d*) *$", query)
    iematch = re.match(r"^ie: +(\d*.\d*) *$", query)
    oematch = re.match(r"^oe: +(\d*.\d*) *$", query)
    oermatch = re.match(r"^oer: +(\d*.\d*) *$", query)
    ie = oe = 0.001
    oer = 0
    # overide values with file values
    if pmatch:
        ie = oe = float(pmatch.group(1))
    if iematch:
        ie = float(iematch.group(1))
    if oematch:
        oe = float(oematch.group(1))
    if oermatch:
        oer = float(oermatch.group(1))
    #overide those with command line values
    if args.prec != None:
        ie = oe = args.prec
    if args.input_epsilon != None:
        ie = args.input_epsilon
    if args.output_epsilon != None:
        oe = args.output_epsilon
    if args.output_epsilon != None:
        oe = args.output_epsilon
    if args.relative_input_epsilon != None:
        oer = args.relative_input_epsilon



    # vars
    lines = [line.strip() for line in query.splitlines() if line.strip()!=''and line.strip()[0] != '#']
    try:
        start = lines.index("var:")
    except:
        print("Malformed query file, no var section: {}".format(args.query_file))
        sys.exit(-1)
    var_lines = list()
    names = set()
    for line in lines[start+1:]:
        if ':' in line:
            break
        match = re.search(r"(\[[^,]+, *[^\]]+\]) *([^;]+)", line)
        if match:
            val = match.group(1)
            name = match.group(2)
            if name in SYMBOLIC_CONSTS:
                iu.log(1, lambda :(iu.yellow("Warning: ") +
                                   "disregarding assignment to {}".format(name)))
                continue
            if name in names:
                print("Duplicate variable definition {}".format(name))
                sys.exit(-1)
            names.add(name)
        else:
            print("Malformed query file, imporoper var: {}".format(line))
            sys.exit(-1)
        var_lines.append("{} = {};".format(name, val))
    var_lines = '\n'.join(var_lines)

    # cost
    try:
        start = lines.index("cost:")
    except:
        print("Malformed query file, no cost section: {}".format(args.query_file))
        sys.exit(-1)
    function = list()
    part = ""
    for line in lines[start+1:]:
        if ':' in line:
            break
        if ';' in line:
            function.append("(" + part + line.replace(';','') + ")")
            part = ""
        else:
            part += line.strip()
    if part != "":
        function.append("({})".format(part))
    function = '+'.join(function)

    if args.dreal:
        function = "-({})".format(function)

    # constraints
    try:
        start = lines.index("ctr:")
    except:
        start = False

    constraints = list()
    if start:
        for line in lines[start+1:]:
            if ':' in line:
                break
            constraints.append(line)
        print("Gelpia does not currently handle constraints")
        sys.exit(-1)

    constraints = '\n'.join(constraints)

    # combining and parsing
    reformatted_query = '\n'.join((var_lines, constraints, function))

    return (args, reformatted_query, [ie, oe, oer])


def finish_parsing_args(args, function, epsilons):
    if args.debug or args.verbose:
        iu.set_log_level(max(1, args.verbose))

    exp = parse_function(function)
    exp, inputs, assigns = lift_inputs_and_assigns(exp)
    exp = simplify(exp, inputs, assigns)

    rev_diff = reverse_diff(exp, inputs, assigns)
    rev_diff = single_assignment(rev_diff, inputs, assigns)
    rev_diff = simplify(rev_diff, inputs, assigns)

    r, rev_diff, consts = lift_consts(rev_diff, inputs, assigns)
    rev_diff = simplify(rev_diff, inputs, assigns, consts)
    _, new_assigns, _ = dead_removal(rev_diff, inputs, assigns, consts)

    i, interp_exp, consts = lift_consts(exp, inputs, assigns, consts)
    interp_exp = simplify(interp_exp, inputs, assigns, consts)
    i, interp_exp, consts = lift_consts(exp, inputs, assigns, consts)

    rust_func, new_inputs, new_consts = to_rust(rev_diff,
                                                inputs, new_assigns, consts)

    interp_func = to_interp(interp_exp, inputs, assigns, consts)
    human_readable = lambda : flatten(rev_diff, inputs, assigns, consts, True)

    return {"input_epsilon"      : epsilons[0],
            "output_epsilon"     : epsilons[1],
            "rel_output_epsilon" : epsilons[2],
            "inputs"             : new_inputs,
            "constants"          : '|'.join(new_consts.values()),
            "rust_function"      : rust_func,
            "interp_function"    : interp_func,
            "expression"         : exp,
            "human_readable"     : human_readable,
            "debug"              : args.debug,
            "timeout"            : args.timeout,
            "grace"              : args.grace,
            "update"             : args.update,
            "logfile"            : args.logging,
            "dreal"              : args.dreal,
            "fptaylor"           : args.fptaylor,
            "iters"              : args.maxiters,
            "seed"               : args.seed,
            "concurrency"        : args.concurrency,
            "np"                 : args.np,}
