"""Exploratory L31/L41 analysis consuming only captured actual-vLLM tensors."""
from __future__ import annotations

from pathlib import Path
import csv

from .run_state import read_jsonl, atomic_write_json, write_jsonl
from .capture_states import load_rank_records, load_raw_logits
from .residual_ledger import index_capture_records, extract_layer_tensors
from .math_utils import cosine, fisher_inner, fisher_quadratic_kl, sample_bootstrap_ci
from .objective_holdout import _nmse

COMPARISONS = ((31, 'O2_M', 'O3_full'), (41, 'O2_A', 'O4'))


def sample_statistics(rows: list[dict], key: str, *, sample_ids: list[str],
                      prefix_lengths=(8,32,64,128)) -> dict:
    groups = {}
    for row in rows:
        groups.setdefault(row['calibration_sample_id'], []).append(row)
    if len(sample_ids)<2 or len(set(sample_ids))!=len(sample_ids) or set(groups)!=set(sample_ids) or any(sorted(r['prefix_length_j'] for r in g)!=sorted(prefix_lengths) for g in groups.values()):
        raise RuntimeError('paired samples or state coverage differ from frozen manifest')
    means = {sid: sum(r[key] for r in group) / len(prefix_lengths) for sid, group in groups.items()}
    sources = {sid: group[0]['source'] for sid, group in groups.items()}
    values = list(means.values())
    result = sample_bootstrap_ci(values, seed=20260909, n_boot=10000)
    result['positive_samples'] = sum(v > 0 for v in values)
    result['negative_samples'] = sum(v < 0 for v in values)
    result['sample_count'] = len(values)
    result['leave_one_sample_out_mean_range'] = [min((sum(values)-v)/(len(values)-1) for v in values), max((sum(values)-v)/(len(values)-1) for v in values)]
    result['sample_means'] = means
    result['sources'] = {source: sample_bootstrap_ci([v for sid,v in means.items() if sources[sid]==source],
                                                   seed=20260909, n_boot=10000)
                         for source in ('wikitext2','s1k_original')}
    return result


def run_mechanism_review(run_root: Path) -> dict:
    root = run_root / '60_objective/actual_holdout'
    out = run_root / '60_objective/path_mechanism_audit/mechanism'
    out.mkdir(parents=True, exist_ok=True)
    states = read_jsonl(root / 'cohort.jsonl')
    summaries, all_rows, depth_rows = {}, [], []
    for layer, base_loss, cand_loss in COMPARISONS:
        base_label, cand_label = f'L{layer}_{base_loss}', f'L{layer}_{cand_loss}'
        base = {r['sample_key']:r for r in read_jsonl(root/base_label/'state_metrics.jsonl')}
        cand = {r['sample_key']:r for r in read_jsonl(root/cand_label/'state_metrics.jsonl')}
        e1 = {r['sample_key']:r for r in read_jsonl(root/'E1/state_metrics.jsonl') if r['layer']==layer}
        if set(base) != set(cand) or set(base) != {s['sample_key'] for s in states}:
            raise RuntimeError('comparison state coverage mismatch')
        rows = []
        for meta in states:
            key, di = meta['sample_key'], meta['decode_index']
            b, c = base[key], cand[key]
            row = {k:meta[k] for k in ('sample_key','calibration_sample_id','source','prefix_length_j')}
            row.update(layer=layer, baseline=base_loss, candidate=cand_loss,
                       kl_gain=b['final_logit_kl']-c['final_logit_kl'],
                       candidate_vs_e1_kl_gain=e1[key]['final_logit_kl']-c['final_logit_kl'],
                       cumulative_nmse_gain=b['cumulative_residual_nmse']-c['cumulative_residual_nmse'],
                       final_hidden_nmse_gain=b['final_hidden_nmse']-c['final_hidden_nmse'],
                       positive_G_gain=b['positive_G']-c['positive_G'])
            # Both models are separately intervened with the SAME E0 router logits.
            # This is an interaction contrast, not an additive mediation fraction.
            frozen_b = b['final_logit_kl']-b['router_causal_contribution']
            frozen_c = c['final_logit_kl']-c['router_causal_contribution']
            row['kl_gain_after_e0_router_freeze'] = frozen_b-frozen_c
            row['router_interaction_contrast'] = row['kl_gain']-row['kl_gain_after_e0_router_freeze']
            row['router_full_kl_gain'] = b['router_full_kl']-c['router_full_kl']
            row['router_topk_total_gain'] = b['router_topk_total']-c['router_topk_total']
            tensors = []
            for label, variant in [('E0','E0'),(base_label,'E1'),(cand_label,'E1')]:
                records = load_rank_records(root/label/'hooks', variant, key, 0)
                tensors.append(extract_layer_tensors(index_capture_records(records), sample_key=key, decode_index=di))
            t0, tb, tc = tensors
            for boundary in range(49):
                nmse_b = _nmse(tb['R'][boundary], t0['R'][boundary])
                nmse_c = _nmse(tc['R'][boundary], t0['R'][boundary])
                depth_rows.append({**{k:row[k] for k in ('sample_key','calibration_sample_id','source','prefix_length_j','layer','baseline','candidate')},
                                   'boundary':boundary, 'baseline_nmse':nmse_b,'candidate_nmse':nmse_c,'nmse_gain':nmse_b-nmse_c})
            for name, ref, old, new in [('selected_output', t0['R'][layer+1],tb['R'][layer+1],tc['R'][layer+1]),
                                       ('final_residual',t0['R'][48],tb['R'][48],tc['R'][48])]:
                error = old.double()-ref.double()
                update = new.double()-old.double()
                row[name+'_error_update_cosine'] = cosine(error,update)
                row[name+'_error_energy_change'] = float((new.double()-ref.double()).square().sum()-error.square().sum())
                row[name+'_update_energy'] = float(update.square().sum())
                row[name+'_cross_term'] = float(2*(error*update).sum())
            l0 = load_raw_logits(root/'E0/raw_logits','E0',key,di)
            lb = load_raw_logits(root/base_label/'raw_logits','E1',key,di)
            lc = load_raw_logits(root/cand_label/'raw_logits','E1',key,di)
            vb, update = lb.double()-l0.double(), lc.double()-lb.double()
            row['fisher_baseline'] = fisher_quadratic_kl(l0,vb)
            row['fisher_candidate'] = fisher_quadratic_kl(l0,lc.double()-l0.double())
            row['fisher_update_self'] = .5*fisher_inner(l0,update,update)
            row['fisher_update_cross'] = fisher_inner(l0,vb,update)
            row['fisher_baseline_abs_error'] = abs(row['fisher_baseline']-b['final_logit_kl'])
            row['fisher_candidate_abs_error'] = abs(row['fisher_candidate']-c['final_logit_kl'])
            rows.append(row)
        label = f'L{layer}_{cand_loss}_vs_{base_loss}'
        metrics = ['kl_gain','candidate_vs_e1_kl_gain','cumulative_nmse_gain','final_hidden_nmse_gain',
                   'positive_G_gain','kl_gain_after_e0_router_freeze','router_interaction_contrast',
                   'router_full_kl_gain','router_topk_total_gain','fisher_baseline_abs_error','fisher_candidate_abs_error']
        summaries[label] = {key:sample_statistics(rows,key,sample_ids=sorted({s['calibration_sample_id'] for s in states})) for key in metrics}
        all_rows.extend(rows)
    write_jsonl(out/'paired_state_analysis.jsonl',all_rows)
    write_jsonl(out/'depth_profiles.jsonl',depth_rows)
    sample_rows = []
    for label, metrics in summaries.items():
        for sid in metrics['kl_gain']['sample_means']:
            sample_rows.append({'comparison':label,'sample_id':sid,
                               **{key:value['sample_means'][sid] for key,value in metrics.items()}})
    with (out/'sample_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(sample_rows[0]));writer.writeheader();writer.writerows(sample_rows)
    atomic_write_json(out/'summary.json',summaries)
    lines=['# L31 / L41 机制复盘','',
           '本报告只使用已有 actual-vLLM 采集。16 个验证样本已用于选择研究方向，以下是探索性配对分析，不是新的独立确认。',
           '正的 gain 表示该指标下降；四个 state 先在 sample 内平均。', '',
           '| 对照 | KL 改善 [95% CI] | 改善样本 | 两侧都冻结 E0 Router 后的 KL 改善 | 累计 NMSE 改善 |',
           '|---|---|---|---|---|']
    for label,m in summaries.items():
        kl=m['kl_gain']; fr=m['kl_gain_after_e0_router_freeze']; nm=m['cumulative_nmse_gain']
        lines.append(f"| {label} | {kl['mean']:.6g} [{kl['ci95_lo']:.6g}, {kl['ci95_hi']:.6g}] | {kl['positive_samples']}/16 | {fr['mean']:.6g} [{fr['ci95_lo']:.6g}, {fr['ci95_hi']:.6g}] | {nm['mean']:.6g} |")
    for label,m in summaries.items():
        kl=m['kl_gain']
        lines += ['',f'## {label} 来源差异','']
        for source,stat in kl['sources'].items():
            lines.append(f"- {source}: KL gain={stat['mean']:.6g}, CI=[{stat['ci95_lo']:.6g}, {stat['ci95_hi']:.6g}]")
        lines.append(f"- 去掉任意一个样本后的平均 KL gain 范围：{kl['leave_one_sample_out_mean_range']}")
    lines += ['', '## 解释边界', '',
              '- Router freeze 对照测量当前 predictor 上的路由干预交互；它不是 Router 贡献百分比，也不能把非线性收益强行相加。',
              '- depth_profiles.jsonl 定位改善/恶化在后续哪些层出现，不能仅凭相关性断言因果。',
              '- Fisher 分解只作局部二阶描述，同时提供相对 exact KL 的绝对误差；正式输出结论始终依据 exact KL。',
              '- 路径一致性审计完成前，不能把这些旧 checkpoint 的差异归因于理想定义下的 O2/O3/O4。']
    (out/'MECHANISM_REVIEW.md').write_text('\n'.join(lines)+'\n')
    return summaries
