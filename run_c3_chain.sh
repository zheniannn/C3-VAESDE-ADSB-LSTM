set -e
VPY=/home/ian/.venvs/venv/bin/python
CFG=configs/lstm_default.yaml
echo "### C3 TRAIN ###";  $VPY scripts/run_train_lstm.py --config $CFG
echo "### C3 SCORE ###";  $VPY scripts/run_score_lstm.py --config $CFG
echo "### C3 STRESS ###"; $VPY scripts/run_stress_test_lstm.py --config $CFG
echo "### C3 CHAIN COMPLETE ###"
