# -*- coding: utf-8 -*-
"""
@author: Theo-eBrown

Written for use with .exe file
"""
import scrollthescroll
from os import path

file_path = path.abspath(path.join(""))+"\\" #path to pyinstaller dist

scrollthescroll.package_path = file_path

p = scrollthescroll.Prototype()

p.run(file_path=file_path)