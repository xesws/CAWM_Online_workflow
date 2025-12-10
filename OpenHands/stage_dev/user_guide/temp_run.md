poetry run python -m evaluation.awm.cli \
    --llm-config llm.kimi-k2 \
    --dataset "princeton-nlp/SWE-bench_Lite" \
    --repo-filter "" \
    --output-dir "evaluation/evaluation_outputs/awm_swebench_lite" \
    --max-iterations 100 \
    --truncation 30000 \
    --induction-trigger 1 