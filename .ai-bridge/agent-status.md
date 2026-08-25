# Agent Status

Task 0-13 audited, implemented, and verified through PASS.

Current stop point: Task 13 complete. Do not start Task 14-19 formal 25-run training matrix automatically.

Key Task 13 evidence:
- Full e2e reconstruction tests: `CUDA_VISIBLE_DEVICES=0 conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction -q` -> `95 passed, 15 warnings`.
- Old 8B baseline tests: `conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/test_*.py -q` -> `68 passed`.
- Task 13 I converted TP2 smoke: `/tmp/hif4_task13_smoke/I_E1_direct_converted_tp2_bf16.json` -> `passed=true`, `resolved_kv_cache_dtype=bfloat16`, `sidecar_effective=true`, trace `sidecar_load=2`, `dense_apply=288`, `moe_apply=2`.
- Task 13 J Native E0 TP2 smoke: `/tmp/hif4_task13_smoke/J_native_tp2_bf16.json` -> `passed=true`, `resolved_linear_backend=emulation`, `resolved_moe_backend=emulation`, `resolved_kv_cache_dtype=bfloat16`, `no_marlin=true`.
- Task 13 K TP parity diagnostic: `/tmp/hif4_task13_tp_parity/summary_top100.json` and `/tmp/hif4_task13_tp_parity/layer0_puncture.json`; evidence supports Case A, so greedy generation exact-match remains report-only under the revised numerical parity gate.

Updated reports:
- `Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_IMPLEMENTATION_PROGRESS.md`
- `Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_IMPLEMENTATION_STOP_REASON.md`
- `Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_TP_PARITY_DIAGNOSTIC.md`

Review notes:
- The converted vLLM sidecar path is now captured at unquantized linear/MoE method construction and passed explicitly at runtime; this avoids depending on execution-time thread-local vLLM config in worker processes.
- Qwen3-MoE shared experts remain handled by the vLLM runner; the HiF4 MoE runtime returns only routed expert output.
- `max_num_batched_tokens=512` is used only in the converted smoke to keep vLLM profile scale aligned with the 512-token Task 13 smoke target; TP2, BF16 KV, prompt batch, and generated token gate are unchanged.
