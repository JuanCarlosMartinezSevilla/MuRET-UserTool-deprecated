for f in $(seq 0 4); do
  for fusion in "cnn" "rnn" "softmax"; do
    for a in $(seq 1 5); do
      for b in $(seq 0 5); do


        echo "##### $f $fusion $a $b #####"

        python -u experiment.py -tr /media/jcalvo/Data/Datasets/MuRET/5-CV/Capitan/train_gt_fold$f.dat  -val /media/jcalvo/Data/Datasets/MuRET/5-CV/Capitan/val_gt_fold$f.dat -ts /media/jcalvo/Data/Datasets/MuRET/5-CV/Capitan/test_gt_fold$f.dat -a $a -b $b -f $fusion > LOG_$f-$fusion-$a-$b.log

      done
    done
  done
done

