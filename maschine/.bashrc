# Sample .bashrc for SuSE Linux
# Copyright (c) SuSE GmbH Nuernberg

# There are 3 different types of shells in bash: the login shell, normal shell
# and interactive shell. Login shells read ~/.profile and interactive shells
# read ~/.bashrc; in our setup, /etc/profile sources ~/.bashrc - thus all
# settings made here will also take effect in a login shell.
#
# NOTE: It is recommended to make language settings in ~/.profile rather than
# here, since multilingual X sessions would not work properly if LANG is over-
# ridden in every subshell.

# Some applications read the EDITOR variable to determine your favourite text
# editor. So uncomment the line below and enter the editor of your choice :-)
#export EDITOR=/usr/bin/vim
#export EDITOR=/usr/bin/mcedit

# For some news readers it makes sense to specify the NEWSSERVER variable here
#export NEWSSERVER=your.news.server

# If you want to use a Palm device with Linux, uncomment the two lines below.
# For some (older) Palm Pilots, you might need to set a lower baud rate
# e.g. 57600 or 38400; lowest is 9600 (very slow!)
#
#export PILOTPORT=/dev/pilot
#export PILOTRATE=115200

# Load pyenv automatically by appending
# the following to 
# ~/.bash_profile if it exists, otherwise ~/.profile (for login shells)
# and ~/.bashrc (for interactive shells) :

#--------------------------------
# Default work folder
#--------------------------------

# export USERDIR=/tp5-peeters/bt712347
export USERDIR=~/Documents
export WORKDIR=$USERDIR/Masterthesis-Parallel-Electric-Field

#--------------------------------
# GKW paths
#--------------------------------

export GKW_HOME=$WORKDIR/gkw
export GKW_RUN=$GKW_HOME/run
export GKW_EXEC=$GKW_RUN/gkw.x
export PATH=$GKW_HOME/scripts:$PATH

#--------------------------------
# PYENV
#--------------------------------

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Restart your shell for the changes to take effect.

# Load pyenv-virtualenv automatically by adding
# the following to ~/.bashrc:

eval "$(pyenv virtualenv-init -)"

#--------------------------------
# Change bash prompt design
#--------------------------------

export PS1="\[\e[01;32m\]\u \W\[\e[m\]\n\[\e[01;34m\]>\[\e[m\]\[\e[01;31m\]>\[\e[m\]\[\e[11;33m\]>\[\e[m\] "

source /home/btpp/btp00000/shared/source_me_to_have_them_all.sh

# Source alias file
test -s ~/.alias && . ~/.alias || true
