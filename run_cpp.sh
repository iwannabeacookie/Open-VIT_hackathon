source params.sh

if [ ! -d $CPP_OUT_FOLDER ]; then
    mkdir $CPP_OUT_FOLDER
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

if [ ! -f "bin/vit.exe" ]; then
    echo Error: missing bin/vit.exe file!
    echo Run compile.sh script
    exit 1
fi



if [ ! -f $MEASURES_FOLDER/cpp.csv ]; then
    touch $MEASURES_FOLDER/cpp.csv
fi
echo "batch_size;model_depth;load_cvit_time;load_cpic_time;foreward_time[];store_cprd_time" >$MEASURES_FOLDER/cpp.csv
for ((i=0; i<$DTASET_DIM; i++)); do
    ./bin/vit.exe $MODEL_PATH $DTASET_FOLDER/pic_$i.cpic $CPP_OUT_FOLDER/prd_$i.cprd $MEASURES_FOLDER/cpp.csv
done
echo vit executed

python3 scripts/analyze_time_measures.py $MEASURES_FOLDER/cpp.csv $MEASURES_FOLDER/cpp_summary.txt
