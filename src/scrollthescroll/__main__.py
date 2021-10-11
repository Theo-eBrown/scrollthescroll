# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:37:50 2021

@author: Theo-eBrown
"""

import __init__
import sys

if __name__ == "__main__":
    if sys.argv[-1] == "-h":
        #help for running
        pass
    else:
        try:
            __init__.Prototype().run(sys.argv[1],int(sys.argv[2]))
        except IndexError:
            raise SyntaxError("Incorrect syntax, for help run 'scrollthescroll -h'")
