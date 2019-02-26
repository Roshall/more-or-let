#! /usr/bin/env bash
# Copyright 2017 Sean Robertson

set -e

source runsteps/xx_utility_funcs.sh
train_cmd="run.pl"
logdir=exp/log/train
cmd=$train_cmd
model_conf=conf/model.conf
verbose=0
model_formatter=
source utils/parse_options.sh
FEAT_NAMES="kaldi"

if $PREPROCESS_ON_BATCH ; then
  num_feats=41
else
  num_feats=123
fi

iecho "Adam training"

for feat in ${FEAT_NAMES} ; do
  source runsteps/08a_train_ctc.sh \
    --training-stage adam \
    --verbose ${verbose} \
    ${feat}_${num_feats} data/${num_feats} exp
done

iecho "Done Adam training"
