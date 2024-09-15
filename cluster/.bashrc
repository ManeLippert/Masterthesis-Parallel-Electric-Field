# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Activate extglobs
shopt -s extglob

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

#--------------------------------
# Change bash prompt design
#--------------------------------

export PS1="\[\e[01;33m\]\u \W\[\e[m\]\n\[\e[01;32m\]>\[\e[m\]\[\e[01;31m\]>\[\e[m\]\[\e[11;34m\]>\[\e[m\] "

#--------------------------------
# Load modules for GKW
#--------------------------------

# For emil
module load inteloneapi/mpi/2021.8.0 &> /dev/null 
module load inteloneapi/compiler/2022.1.0  &> /dev/null
module load inteloneapi/compiler-rt/2021.3.0  &> /dev/null
module load inteloneapi/tbb/2021.3.0  &> /dev/null
module load hdf5/1.12.1  &> /dev/null
module load fftw/3.3.10 &> /dev/null

# For festus
#module load OneAPI_2024.2.1 &> /dev/null
#module load openmpi/5.0.3 &> /dev/null
#module load phdf5/1.14.4-2 &> /dev/null
#module load fftw/3.3.10 &> /dev/null

#--------------------------------
# Default work folder
#--------------------------------

export USERDIR=/scratch/bt712347
export WORKDIR=$USERDIR
export SCRIPTS_PATH=$WORKDIR/scripts

# cd into workdir after login
cd $WORKDIR

#--------------------------------
# GKW paths
#--------------------------------

export GKW_HOME=$WORKDIR/gkw
export GKW_RUN=$GKW_HOME/run
export GKW_EXEC=$GKW_RUN/gkw.x
export PATH=$GKW_HOME/scripts:$PATH

# Source alias file
test -s ~/.alias && . ~/.alias || true