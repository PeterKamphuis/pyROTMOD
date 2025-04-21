# -*- coding: future_fstrings -*-
# This module  contains (eventually all) functions that relate to the 
# logging process of the code. This includes the creation of the log file

from inspect import stack

def linenumber(debug='short'):
    '''get the line number of the print statement in the main.'''
    line = []
    for key in stack():
        if key[1] == 'main.py':
            break
        if key[3] != 'linenumber' and key[3] != 'print_log' and key[3] != '<module>':
            file = key[1].split('/')
            to_add= f"In the function {key[3]} at line {key[2]}"
            if debug == 'long':
                to_add = f"{to_add} in file {file[-1]}."
            else:
                to_add = f"{to_add}."
            line.append(to_add)
    if len(line) > 0:
        if debug == 'long':
            line = ', '.join(line)+f'\n{"":8s}'
        elif debug == 'short':
            line = line[0]+f'\n{"":8s}'
        else:
            line = f'{"":8s}'
    else:
        for key in stack():
            if key[1] == 'main.py':
                line = f"{'('+str(key[2])+')':8s}"
                break
    return line
linenumber.__doc__ =f'''
 NAME:
    linenumber

 PURPOSE:
    get the line number of the print statement in the main. Not sure 
    how well this is currently working.

 CATEGORY:
    log_functions 

 INPUTS:

 OPTIONAL INPUTS:


 OUTPUTS:
    the line number of the print statement

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    Unspecified

 NOTE:
    If debug = True the full stack of the line print will be given, 
    in principle the first debug message in every function should set 
    this to true and later messages not.
    !!!!Not sure whether currently the linenumber is produced due to 
    the restructuring.
'''

def print_log(log_statement,cfg, case = None):
    '''Print statements to log if existent and screen if Request'''
    if cfg is None:
        setattr(cfg,'output',None)
        setattr(cfg.output,'debug',False) 
        setattr(cfg.output,'log',None)
        setattr(cfg.output,'log_directory','./')
        setattr(cfg.output,'verbose',False)

    if case is None:
        case=['main']
    debugging = False
    debug= 'empty'
    if cfg.output.debug:
        trig=False
        if 'ALL' in cfg.output.debug_functions:
            trig = True
        else:
            # get the function  
            for key in stack():
                if key[3] != 'linenumber' and key[3] != 'print_log' and key[3] != '<module>': 
                    current_function= f"{key[3]}"
                    break
            if current_function.lower() in [x.lower() for x in cfg.output.debug_functions]:
                trig=True      
        if trig:
            debugging=True    
            if 'debug_start' in case:
                debug = 'long'
            else:
                debug= 'short'
    if not 'main' in case:
        log_statement = f"{linenumber(debug=debug)}{log_statement}"
    print_statement = False
    if (debugging and ('debug_start' in case or 'debug_add' in case))\
        or 'main' in case:
        if cfg.output.log is None or cfg.output.verbose or 'screen' in case:
            print(log_statement)
        if not cfg.output.log is None:
            with open(f"{cfg.output.log_directory}{cfg.output.log}",'a') as log_file:
                log_file.write(log_statement)

print_log.__doc__ =f'''
 NAME:
    print_log
 PURPOSE:
    Print statements to log if existent and screen if Requested
 CATEGORY:
    log_functions 

 INPUTS:
    log_statement = statement to be printed
    Configuration = Standard FAT Configuration

 OPTIONAL INPUTS:


    screen = False
    also print the statement to the screen

 OUTPUTS:
    line in the log or on the screen

 OPTIONAL OUTPUTS:

 PROCEDURES CALLED:
    linenumber, .write

 NOTE:
    If the log is None messages are printed to the screen.
    This is useful for testing functions.
'''
