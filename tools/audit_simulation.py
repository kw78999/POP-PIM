"""Independently audit simulation outputs against their raw 24-byte command traces.

Checks file hashes, command/row conservation, scorer, selector and host-transfer work, two-phase K->V
reads against the V-keep masks, all-bank (REFab) refresh timing, and per-bank (REFsb) timing, debt
window and round rule. Simulator counters are only compared with values re-derived from the trace.
Command codes: 1 ACT, 2 PRE, 3 PREA, 4 RD/RDA, 5 WR/WRA, 6 REFab, 7 other, 8 REFsb.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import BUILD, ROOT, save, sha

PS = 769
PC_MASKS = (0x5555555555555555, 0xaaaaaaaaaaaaaaaa)


def timing():
    """DRAM timing values checked by the build's exact-cycle boundary tests."""
    single = json.loads((BUILD / 'validation/refresh_single_bank.jsonl').read_text().splitlines()[-1])
    allbank = json.loads((BUILD / 'validation/refresh.jsonl').read_text().splitlines()[0])
    assert single['phase20_refresh_boundaries_passed'] and allbank['timing_boundaries_passed']
    return dict(nRP=single['nRP'], nRC=single['nRC'], nRFCSB=single['nRFCSB'], nRREFD=single['nRREFD'],
                nREFI=single['nREFI'], nRFC=allbank['nRFC'])


def config_value(text, key):
    for line in text.splitlines():
        if line.strip().startswith(key + ':'):
            return line.split(':', 1)[1].strip()
    return None


def _gap(stamps, previous):
    index = np.searchsorted(previous, stamps, side='right') - 1
    valid = index >= 0
    return int((stamps[valid] - previous[index[valid]]).min()) if valid.any() else 10**9


def audit_work(record, m):
    """Conservation checks that need only the metrics."""
    n = 32 * record['context']
    assert m['row_hits'] + m['row_misses'] + m['row_conflicts'] == m['RD']
    assert m['RD'] == m['read_requests'] == m['callbacks'] and m['read_bytes'] == m['read_requests'] * 32
    assert m['score_count'] == n and m['replicas_per_bank'] == 1
    if record['mode'] == 'full':
        assert m['attention_output_ps'] >= max(m['last_score_ps'], m['av_ready_ps'])
    else:
        assert m['attention_output_ps'] == 0
    if not record['design'].startswith('POP_'):
        assert m['exact_finite_tile'] == 1 and m['scorer_latency_added_input_cost'] == 0
        schedule = json.loads((ROOT / 'data/calibration/tile_schedules.json').read_text())[record['design']]
        assert m['tile_service_cycles'] == schedule['ready'][-1] + 1
        return
    variants = json.loads((ROOT / 'data/replay/variants/manifest.json').read_text())
    from .run_simulation import VARIANTS
    source, budget, mask, compact = VARIANTS[record['variant']]
    assert m['rtl_router_enqueues'] == m['rtl_router_dequeues'] == n and m['rtl_operands'] == 64 * record['context']
    assert m['read_requests'] == m['metadata_read_requests'] + m['selected_kv_read_requests']
    if record['mode'] != 'full':
        return
    assert m['selected_union_tokens'] == variants['inputs'][source]['union_tokens']
    if mask:
        kept = variants['vkeep'][mask]['kept_v_tokens']
        assert m['phase21_two_phase'] == 1 and m['phase21_kept_v_tokens'] == kept
        assert m['phase21_selected_k_reads'] == 8 * m['selected_union_tokens']
        assert m['phase21_selected_v_reads'] == 8 * kept
        assert m['selected_kv_read_requests'] == m['phase21_selected_k_reads'] + m['phase21_selected_v_reads']
    else:
        assert m['phase21_two_phase'] == 0 and m['selected_kv_read_requests'] == 16 * m['selected_union_tokens']
    if m.get('offload'):
        assert m['selector_ingested'] == 0 and m['score_output_packets'] == n and m['score_payload_bytes'] == 2 * n
        if compact:
            assert m['phase21_shared_export'] == 1 and m['score_wire_bytes'] == 2 * (n // 4)
            assert m['phase21_index_return_bytes'] == 8 * record['context'] // 8
        else:
            assert m['phase21_shared_export'] == 0 and m['score_wire_bytes'] == 8 * n
            groups = 32 if source == 'base' else 8
            assert m['phase21_index_return_bytes'] == groups * budget * 4
        assert m['score_output_fifo_peak_packets'] <= m['score_output_fifo_capacity_packets'] == 64
        assert m['dma_queue_peak'] <= m['dma_queue_depth'] == 16
        assert m['external_indices_ready_ps'] == (m['score_transfer_last_ps'] + m['external_topk_service_ps']
                                                  + m['index_transfer_ps'] + m['sync_ps'])
        assert m['topk_ready_ps'] >= m['external_indices_ready_ps']
    else:
        assert m['selector_ingested'] == n and m['selector_returned'] == 32 * budget


def audit_refsb(t, mask, code, m, config, T):
    """Per-bank refresh: timing to PRE/ACT/REFsb, debt within [-U, +P], rank spacing and round rule."""
    P, U = int(config_value(config, 'refresh_postpone')), int(config_value(config, 'refresh_pullin'))
    round_rule = config_value(config, 'refresh_round_rule') == 'true'
    cycles = t // PS
    end = m['global_end_ps'] // PS
    selected = code == 8
    rmask = mask[selected]
    assert np.all((rmask & (rmask - np.uint64(1))) == 0), 'REFsb must target exactly one bank'
    rbank = np.log2(rmask.astype(np.float64)).astype(np.int64)
    rcyc = cycles[selected]
    worst = dict(max_debt=-10**9, min_debt_after_refresh=10**9)
    for b in range(64):
        bit = np.uint64(1) << np.uint64(b)
        refs = rcyc[rbank == b]
        assert len(refs) > 0, f'bank {b} never refreshed'
        pres = cycles[((code == 2) & (mask == bit)) | ((code == 3) & ((mask & bit) != 0))]
        acts = cycles[(code == 1) & (mask == bit)]
        to_act = _gap(acts, refs)
        same = int(np.diff(refs).min()) if len(refs) > 1 else 10**9
        assert _gap(refs, pres) >= T['nRP'] and _gap(refs, acts) >= T['nRC'], f'bank {b}: REFsb issued too early'
        assert to_act >= T['nRFCSB'] and same >= T['nRFCSB'], f'bank {b}: nRFCSB violated'
        first = T['nREFI'] * ((b >> 1) + 1) // 32       # staggered obligations, as in the controller
        dues = np.arange(first, end + 1, T['nREFI'])
        debt_at_due = np.arange(1, len(dues) + 1) - np.searchsorted(refs, dues, side='left')
        debt_after_ref = np.searchsorted(dues, refs, side='right') - np.arange(1, len(refs) + 1)
        assert debt_at_due.max(initial=0) <= P, f'bank {b}: refresh postponed beyond {P} tREFI'
        assert debt_after_ref.min() >= -U, f'bank {b}: refresh pulled in beyond {U} tREFI'
        assert -U <= len(dues) - len(refs) <= P
        worst['max_debt'] = max(worst['max_debt'], int(debt_at_due.max(initial=0)))
        worst['min_debt_after_refresh'] = min(worst['min_debt_after_refresh'], int(debt_after_ref.min()))
    for pc in range(2):
        for rank in range(2):
            sel = ((rbank & 1) == pc) & (((rbank >> 3) & 1) == rank)
            gaps = np.diff(rcyc[sel])[rbank[sel][1:] != rbank[sel][:-1]]
            assert not len(gaps) or gaps.min() >= T['nRREFD'], 'nRREFD violated'
    if round_rule:
        for pc in range(2):
            counts = np.zeros(64, dtype=np.int64)
            members = [b for b in range(64) if b & 1 == pc]
            for b in rbank[(rbank & 1) == pc]:
                counts[b] += 1
                assert counts[members].max() - counts[members].min() <= 1, 'round rule violated'
    assert worst['max_debt'] <= m['phase20_max_debt'] and m['phase20_debt_violations'] == 0
    return dict(round_rule=round_rule, postpone=P, pullin=U, **worst)


def audit(record, folder, T=None, check_binary=True):
    """Audit one case folder. Returns a summary dict; raises AssertionError on any violation."""
    T = T or timing()
    folder = Path(folder)
    if check_binary:
        exe = BUILD / 'bin' / ('poppim_pop' if record['design'].startswith('POP_') else 'poppim_dense')
        assert sha(exe) == record['binary_sha256'], 'executable differs from the measured run'
        assert sha(BUILD / 'ramulator/lib/libramulator.so') == record['library_sha256'], 'library differs'
    for name, key in [('config.yaml', 'config_sha256'), ('metrics.json', 'metrics_sha256'), ('command_trace.bin', 'trace_sha256')]:
        assert sha(folder / name) == record[key], f'{folder.name}: {name} hash mismatch'
    for name, value in record['input_sha256'].items():
        path = ROOT / name if (ROOT / name).exists() else None
        if path is not None:
            assert sha(path) == value, f'{name}: input hash mismatch'
    m = json.loads((folder / 'metrics.json').read_text())
    assert m == record['metrics'], 'recorded metrics differ from metrics.json'
    config = (folder / 'config.yaml').read_text()
    policy = config_value(config, 'refresh_policy')
    manager = config.split('RefreshManager:')[1].split('impl:')[1].split()[0]
    path = folder / 'command_trace.bin'
    assert path.stat().st_size % 24 == 0 and path.stat().st_size > 0
    trace = np.memmap(path, dtype='<u8', mode='r').reshape(-1, 3)
    t, mask, code = trace[:, 0].astype(np.int64), trace[:, 1], trace[:, 2].astype(np.int64)
    assert np.all(t[1:] >= t[:-1]) and np.all(t % PS == 0), 'command timestamps out of order or off-clock'
    assert set(np.unique(code).tolist()) <= {1, 2, 3, 4, 6, 8}, 'unexpected command code'
    c = np.bincount(code, minlength=9)
    assert c[1] == m['ACT'] and c[2] + c[3] == m['PRE'] and c[3] == m['PREA'] and c[6] == m['REF']
    assert c[4] == m['RD'] and int(t[-1]) <= m['global_end_ps'] + PS
    audit_work(record, m)
    if manager == 'AllBankHBM3':
        assert c[6] > 0 and c[8] == 0
    elif policy == 'pim_pb':
        assert c[6] == 0 and c[8] == m['REFsb'] > 0
    else:
        assert c[6] == 0 and c[8] == 0
    pre_all = mask[code == 3]
    assert np.all((pre_all == np.uint64(PC_MASKS[0])) | (pre_all == np.uint64(PC_MASKS[1])))
    for pm in PC_MASKS:
        sel = mask == np.uint64(pm)
        refs, preas = t[(code == 6) & sel], t[(code == 3) & sel]
        acts = t[(code == 1) & ((mask & np.uint64(pm)) != 0)]
        assert _gap(refs, preas) >= T['nRP'] * PS and _gap(acts, refs) >= T['nRFC'] * PS, 'REFab timing violation'
    out = dict(name=record['name'], passed=True, command_records=int(len(trace)), REFab=int(c[6]), REFsb=int(c[8]))
    if c[8]:
        out.update(audit_refsb(t, mask, code, m, config, T))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='output directory of tools.run_simulation')
    args = parser.parse_args(argv)
    records = json.loads((args.input_dir / 'runs.json').read_text())
    T = timing()
    checks, skipped = [], []
    for record in records:
        if record.get('trace_retained') is False:
            skipped.append(record['name'])
            continue
        checks.append(audit(record, args.input_dir / record['name'], T))
        print(f"audit passed: {record['name']}", flush=True)
    if not checks:
        raise SystemExit('No retained command traces to audit (runs used --discard-traces)')
    save(args.input_dir / 'audit.json', dict(status='passed', runs=len(checks), skipped_without_trace=skipped,
                                             timing=T, checks=checks))
    print(f'Passed raw command/conservation/refresh audits for {len(checks)} runs'
          + (f' ({len(skipped)} without traces skipped)' if skipped else ''))


if __name__ == '__main__':
    main()
