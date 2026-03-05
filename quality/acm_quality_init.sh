##!/bin/bash
# source it don't execute it !


# bash tuto directory of the script when is sourced
# https://www.baeldung.com/linux/bash-get-location-within-script
ACM_quality=$(realpath $(dirname ${BASH_SOURCE}))

export ACM_root=$ACM_quality/..
export PYTHONPATH=$ACM_root:$PYTHONPATH
export PATH=$ACM_root/scripts/desi/cubic:$ACM_root/scripts/desi/cutsky:$ACM_root/scripts/eft:$PATH
export PATH=$ACM_root/scripts/emc/measurements:$ACM_quality:$PATH
