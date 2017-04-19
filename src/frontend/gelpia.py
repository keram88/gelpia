#!/usr/bin/env python3

import time
import sys
import re
import os
import os.path as path

# Directory names used in this script
full_dir = path.abspath(path.dirname(sys.argv[0])) # Directory for this file
base_dir = path.split(full_dir)[0] # One directory up
src_dir = path.join(base_dir, "src")
bin_dir = path.join(base_dir, "bin")
assert(full_dir == bin_dir)


import ian_utils as iu
import argument_parser as ap


def mk_file_hash(function):
    h  = hash(function)
    h *= hash(time.time())
    h *= os.getpid()
    return hex(h % (1 << 48))[2:] # Trim 0x


def make_X_0(box_list):
    """ Makes a box from the given python structure"""
    box = list()
    for i in range(len(box_list)):
        box.append(box_list[i][0], box_list[i][1])
    return box


def append_to_environ(pathname, addition):
    try:
        os.environ[pathname] = "{}:{}".format(addition,
                                              os.environ[pathname])
    except KeyError:
        os.environ[pathname] = addition


def setup_requirements(base_dir):
    # Add to paths used during runtime for requirements
    path_addition = path.join(base_dir, "requirements/bin")
    append_to_environ("PATH", path_addition)

    ld_lib_addition = path.join(base_dir, "requirements/lib")
    append_to_environ("LD_LIBRARY_PATH", ld_lib_addition)

    lib_addition = path.join(base_dir, "requirements/lib")
    append_to_environ("LIBRARY_PATH", lib_addition)

    c_inc_addition = path.join(base_dir, "requirements/include")
    append_to_environ("C_INCLUDE_PATH", c_inc_addition)

    cplus_inc_addition = path.join(base_dir, "requirements/include")
    append_to_environ("CPLUS_INCLUDE_PATH", cplus_inc_addition)


def var_ordered_output(inputs, lst):
    # Assumes inputs is an ordered dict
    domain = lst[-1]
    domain_string = []
    for i in inputs.keys():
        if i in domain.keys():
            domain_string.append("'{}' : {}".format(i, domain[i]))
    return "[{}, {}]".format(lst[0], "{" + ", ".join(domain_string) + "}")

def main():
    log_level = 0
    setup_requirements(base_dir)

    parsing_start = time.time()
    arg_dict = ap.parse_args()

    if "answer" in arg_dict:
        parsing_end = time.time()
        print("[{}, {{}}]".format(list(arg_dict["answer"])))
        iu.log(log_level, lambda: iu.green("Parsing time: ")+str(parsing_end-parsing_start))
        iu.log(log_level, lambda: iu.green("Solver time: ")+str(0.0))
        return

    # For FPTaylor
    if arg_dict['fptaylor']:
        log_level = 1
    # Add to paths used during runtime for our rust libs
    append_to_environ("PATH", bin_dir)
    rust_ld_lib_addition = path.join(base_dir, ".compiled")
    if arg_dict["debug"]:
        rust_ld_lib_addition += ":" + path.join(base_dir, "src/func/target/debug/")
        rust_ld_lib_addition += ":" + path.join(base_dir, "target/debug/deps")
        # Set debug mode in case of a crash
        append_to_environ("RUST_BACKTRACE", "1")
    else:
        rust_ld_lib_addition += ":" + path.join(base_dir, "src/func/target/release/")
        rust_ld_lib_addition += ":" + path.join(base_dir, "target/release/deps")
    append_to_environ("LD_LIBRARY_PATH", rust_ld_lib_addition)


    # Grab input interval variables, use them for the function translation,
    # and write them out to a rust file
    # start_box, dimensions, variables = parse_box2(arg_dict["input"])
    # import function_to_rust
    # tup = function_to_rust.translate(arg_dict["function"], variables)
    # (function, constants, part) = tup

    inputs = arg_dict['inputs'].values()
    inputs = "|".join(inputs)

    file_id = mk_file_hash(arg_dict["rust_function"])
    function_filename = path.join(src_dir,
                                  "func/src/lib_generated_{}.rs".format(file_id))

    if arg_dict["debug"]:
        executable = path.join(base_dir, 'target/debug/cooperative')
    else:
        executable = path.join(base_dir, 'target/release/cooperative')

    # Log output
    executable_args = ['-c', arg_dict["constants"],
                       '-f', arg_dict["interp_function"],
                       '-i', inputs,
                       "-x", str(arg_dict["input_epsilon"]),
                       "-y", str(arg_dict["output_epsilon"]),
                       "-r", str(arg_dict["rel_output_epsilon"]),
                       "-S", "generated_"+file_id, # Function file suffix
                       "-n", ",".join(arg_dict["inputs"]),
                       "-t", str(arg_dict["timeout"]),
                       "-u", str(arg_dict["update"]),
                       "-M", str(arg_dict["iters"]),
                       "--seed", str(arg_dict["seed"]),
                       "-d" if arg_dict["debug"] else "", # If a debug run
                       "-L" if arg_dict["logfile"] else "",]


    iu.log(1, lambda :iu.cyan("Human Readable:\n") + arg_dict["human_readable"]())
    iu.log(1, lambda :iu.cyan("Interpreted:\n") + arg_dict["interp_function"])
    iu.log(1, lambda :iu.cyan("Rust:\n") + arg_dict["rust_function"])
    iu.log(1, lambda :iu.cyan("Constants:\n") + arg_dict["constants"]+"\n")
    iu.log(1, lambda :iu.cyan("Domain: ") + inputs)
    iu.log(1, lambda :iu.cyan("Variables: ") + ", ".join(b for b in arg_dict["inputs"]))
    iu.log(1, lambda :iu.cyan("Command: ") + ' '.join([executable] + executable_args))

    parsing_end = time.time()

    # Use try so that we can catch control-c easily
    output = ""
    logging = bool(arg_dict["logfile"])
    log_file = arg_dict["logfile"] if type(arg_dict["logfile"]) is str else None

    try:
        with open(function_filename, 'w') as f:
            f.write(arg_dict["rust_function"])

        if log_file:
            with open(log_file, 'w') as f2:
                f2.write("")
                f2.flush()

        start = time.time()
        term_time = None
        if arg_dict["timeout"] != 0:
            if arg_dict["grace"] == 0:
                term_time = start + arg_dict["timeout"]*2
            else:
                term_time = start + arg_dict["grace"]

        iu.log(1, lambda :iu.cyan("Running"))
        for line in iu.run_async(executable, executable_args, term_time, expected_return=-2):
            if line.startswith("lb:"): # Hacky
                if logging:
                    print(line.strip(), file=sys.stderr)
                    if log_file:
                        with open(log_file, 'a') as f2:
                            f2.write(line.strip())
                            f2.write('\n')
                            f2.flush()
            else:
                output += line.strip()
    except KeyboardInterrupt:
        iu.warning("Caught ctrl-C, exiting now")
    finally:
        if not arg_dict["debug"]:
            os.remove(function_filename)
            try:
                os.remove(path.join(base_dir, ".compiled/libfunc_generated_"+file_id+".so"))
            except:
                pass

            try:
                p = path.join(src_dir, "func/target/release/libfunc_generated_"+file_id+".so")
                os.remove(p)
            except:
                pass

            try:
                p = path.join(src_dir, "func/target/release/func_generated_"+file_id+".d")
                os.remove(p)
            except:
                pass
        end = time.time()
    if output:
        try:
            idx = output.find('[')
            restoutput = output[idx:]
            lst = eval(restoutput, {'inf':float('inf')})
            assert(type(lst[-1]) is dict)
            for k in list(lst[-1]):
                if k[0] == "$":
                    del lst[-1][k]
        except:
            print("Error unable to parse rust solver's output: ",output)
            iu.log(log_level, lambda: iu.green("Parsing time: ")+str(parsing_end-parsing_start))
            sys.exit(-1)

        if arg_dict["dreal"]:
            if type(lst[0]) is list:
                lst[0] = reversed(lst[0])
                lst[0] = [-b for b in lst[0]]
            else:
                 lst[0] = -lst[0]

        if not arg_dict["fptaylor"]:
            print(var_ordered_output(arg_dict["inputs"], lst))
        else:
            # We're crashing in compilation for some reason. Will need to
            # investigate
            print("Maximum: {}".format(lst[0][1]) + "\n" +
                  "Minimum: {}".format(lst[0][0]))

    iu.log(log_level, lambda: iu.green("Parsing time: ")+str(parsing_end-parsing_start))
    iu.log(log_level, lambda: iu.green("Solver time: ")+str(end-start))


if __name__ == "__main__":
    main()
