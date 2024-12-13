source params.sh

if [ ! -d $OMP_OUT_FOLDER ]; then
    mkdir $OMP_OUT_FOLDER
fi
if [ ! -d $MEASURES_FOLDER ]; then
    mkdir $MEASURES_FOLDER
fi



if [ ! -f $MODEL_PATH ]; then
    echo Error: missing model $MODEL_PATH!
    exit 1
fi

if [ ! -d $DTASET_FOLDER ]; then
    echo Error: missing dataset $DTASET_FOLDER!
    exit 1
fi

if [ ! -f "omp_bin/vit.exe" ]; then
    echo Error: missing omp_bin/vit.exe file!
    echo Run compile.sh script
    exit 1
fi



if [ ! -f $MEASURES_FOLDER/omp_$NUM_THREADS.csv ]; then
    touch $MEASURES_FOLDER/omp_$NUM_THREADS.csv
fi
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >$MEASURES_FOLDER/omp_$NUM_THREADS.csv

export OMP_NUM_THREADS=$NUM_THREADS
for ((i=0; i<$DTASET_DIM; i++)); do
    ./omp_bin/vit.exe $MODEL_PATH $DTASET_FOLDER/pic_$i.cpic $OMP_OUT_FOLDER/prd_$i.cprd $MEASURES_FOLDER/omp_$NUM_THREADS.csv
done
echo vit executed with $NUM_THREADS threads

python3 scripts/analyze_time_measures.py $MEASURES_FOLDER/omp_$NUM_THREADS.csv $MEASURES_FOLDER/omp_$NUM_THREADS"_summary".txt
