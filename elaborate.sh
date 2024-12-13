source params.sh

if [ ! -d "out_comparison" ]; then
    mkdir "out_comparison"
fi

if [ ! -f "scripts/compare_cpred.py" ]; then
    echo Error: missing scripts/compare_cpred.py file!
    exit 1
fi



# Model Comparison
if [ ! -d $CPP_OUT_FOLDER ]; then
    echo Error: missing $CPP_OUT_FOLDER folder!
    echo Run run_cpp.sh script
    exit 1
fi

if [ ! -d $OMP_OUT_FOLDER ]; then
    echo Error: missing $OMP_OUT_FOLDER folder!
    echo Run run_omp.sh script
    exit 1
fi

for ((i=0; i<$DTASET_DIM; i++)); do
    python3 scripts/compare_cpred.py $CPP_OUT_FOLDER/prd_$i.cprd $OMP_OUT_FOLDER/prd_$i.cprd out_comparison/cpp_vs_omp.txt $CPRD_HIGH_THRESHOLD $CPRD_LOW_THRESHOLD
done
echo $CPP_OUT_FOLDER and $OMP_OUT_FOLDER compared



# Output Comparison Summary
if [ ! -f "scripts/summary_cpred_comparison.py" ]; then
    echo Error: missing scripts/summary_cpred_comparison.py file!
    exit 1
fi
if [ ! -f out_comparison/cpp_vs_omp_summary.txt ]; then
    touch out_comparison/cpp_vs_omp_summary.txt
fi
echo "" >out_comparison/cpp_vs_omp_summary.txt

python3 scripts/summary_cpred_comparison.py out_comparison/cpp_vs_omp.txt out_comparison/cpp_vs_omp_summary.txt
