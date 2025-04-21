# An Introduction to pyROTMOD


Introduction, Installation & Easy use
===========

Introduction
----------------
pyROTMOD is a python based code to model the rotation curves. The code is designed to be flexible and easy to use, allowing users to fit rotation curves with a variety of mass models and data types.
pyROTMOD emulates the GIPSY modules ROTMASS and ROTMOD in a almost independent manner. The fitting and modelling are developed compeletely independent from the GIPSY code with the notable exception of the implementation of the cassertano profiles for arbitrary disks.


Installation
--------

pyROTMOD is a fully python based code hence installation should simply entail

pip install pyROTMOD


Easy use
--------
The easiest way to use pyROTMOD is to use a yaml configuration file. An example with all the defaults can be obtained by typing pyROTMOD print_examples=True, this will then produce a yaml file called pyROTMOD-default.yml
After adapting this file to your specific needs pyROTMOD can be ran by.

pyROTMOD configuration_file=my_conf.yaml

Where my_conf.yaml should be replaced by the name of your configuartion file.

An example configuration file can be obtained through the command 

pyROTMOD print_examples=True

this will then produce a yaml file called pyROTMOD-default.yml