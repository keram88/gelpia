#!/usr/bin/env python3
import subprocess as SP
import argparse as AP
import shlex as SHL
import signal as SIG
import sys as SYS
import time as T
import threading
import queue

class AsyncReader(threading.Thread):
    def __init__(self, fil, q):
        threading.Thread.__init__(self)
        self.q = q;
        self.fil = fil

    def run(self):
        output = "empty"
        while output != "":
            output = self.fil.readline()
            self.q.put(output)

# optional color printing
if SYS.stdout.isatty():
    DO_COLOR = True
else:
    DO_COLOR = False

def use_color_printing():
    global DO_COLOR
    DO_COLOR = True

def use_plain_printing():
    global DO_COLOR
    DO_COLOR = False

def color_text(color_code, text):
    if DO_COLOR:
        return "{}{}{}".format(color_code, text, "\x1b[0m")
    else:
        return text

def black(text):
    return color_text("\x1b[30m", text)

def red(text):
    return color_text("\x1b[31m", text)

def green(text):
    return color_text("\x1b[32m", text)

def yellow(text):
    return color_text("\x1b[33m", text)

def blue(text):
    return color_text("\x1b[34m", text)

def magenta(text):
    return color_text("\x1b[35m", text)

def cyan(text):
    return color_text("\x1b[36m", text)

def white(text):
    return color_text("\x1b[37m", text)




# logging to stdout
LOG_LEVEL = 0

def set_log_level(level):
    global LOG_LEVEL
    LOG_LEVEL = level

def get_log_level():
    return LOG_LEVEL

def log(level, func):
    if (level <= LOG_LEVEL):
        print(func())




# printing to stderr
def error(*objs):
    print(red("ERROR: "), *objs, file=SYS.stderr)

def warning(*objs):
    print(yellow("WARNING: "), *objs, file=SYS.stderr)





# Formatted printing
def box_text(line_list, width,
             top_left="+", top="-", top_right="+",
             left="|", right="|",
             bottom_left="+", bottom="-", bottom_right="+"):
    box = list()
    width_for_top = width - (len(top_left) + len(top_right))
    box.append("{}{}{}".format(top_left,
                               top * (width_for_top // len(top)),
                               top_right))

    width_for_middle = width - (len(left) + len(right))
    format_string = "{}{:"+str(width_for_middle)+"}{}"
    box.extend([format_string.format(left, line, right) for line in line_list])

    width_for_bottom = width - (len(bottom_left) + len(bottom_right))
    box.append("{}{}{}".format(bottom_left,
                               bottom * (width_for_bottom // len(bottom)),
                               bottom_right))

    return box

def comment_block(line_list, width):
    return box_text(line_list, width,
                   "/*","*","*",
                   " * ", "",
                   " *", "*", "*/")




# popen wrappers
def run(cmd, args_list, error_string="An Error has occured", expected_return=0):
    command = [cmd]+args_list
    should_exit = None
    try:
        with SP.Popen(command,
                      stdout=SP.PIPE, stderr=SP.STDOUT) as proc:
            output = proc.stdout.read().decode("utf-8")
            proc.wait()

            if (expected_return != None) and (proc.returncode != expected_return):
                error(error_string)
                error("Return code: {}".format(proc.returncode))
                error("Command used: {}".format(command))
                error("Trace:\n{}".format(output))
                should_exit = proc.returncode
    except KeyboardInterrupt:
        raise
    except:
        error("Unable to run given executable, does it exist?")
        error("executable: {}".format(command))
        SYS.exit(-1)

    if (should_exit != None):
        SYS.exit(should_exit)

    return output

def run_async(cmd, args_list, term_time, procs, error_string="An Error has occured",
              expected_return=0):
    command = ["mpiexec", "-np", str(procs)] + [cmd] + args_list
    should_exit = None
    try:
        with SP.Popen(command, bufsize=1, universal_newlines=True,
                      stdout=SP.PIPE, stderr=SP.STDOUT) as proc:
            start_time = T.time()

            # Asynchronously collect messages
            stdout_q = queue.Queue();
            stdout_r = AsyncReader(proc.stdout, stdout_q);
            stdout_r.start()

            while proc.poll() is None:
                if not stdout_q.empty():
                    line = stdout_q.get()
                    yield line

                # Kill proc if timeout exceeded
                if term_time is not None:
                    if T.time() > term_time:
                        print("Killed")
                        proc.kill()
                T.sleep(0.1)

            # Clear remaining buffered messages
            while stdout_r.is_alive() or not stdout_q.empty():
                if not stdout_q.empty():
                    yield stdout_q.get()

            proc.wait()
            if (expected_return != None) and (proc.returncode not in
                                              (expected_return, -9)):
                error(error_string)
                error("Return code: {}".format(proc.returncode))
                error("Command used: {}".format(command))
                error("Trace:\n{}".format(output))
                should_exit = proc.returncode
    except KeyboardInterrupt:
        raise
    except:
        error("Unable to run given executable, does it exist?")
        error("executable: {}".format(command))
        SYS.exit(-1)

    if (should_exit != None):
        SYS.exit(should_exit)




# function timing
def time_func(function, *args):
    start = T.time()
    ret = function(*args)
    end = T.time()
    return (end-start, ret)




# nicer file parsing for args
class IanArgumentParser(AP.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(IanArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        try:
            for arg in SHL.split(line):
                if not arg.strip():
                    continue
                if arg[0] == '#':
                    break
                yield arg
        except:
            error("Unable to parse argument string")
            error("given string: {}".format(line))
            SYS.exit(-1)
