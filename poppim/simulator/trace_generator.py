"""Phase-level analytical command trace, not Ramulator/cycle-accurate output."""
import csv


def write_trace(path, estimates):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['mode','command','duration_ns','overlap_group','source'])
        writer.writeheader()
        for row in estimates:
            mode = row['mode']
            commands = [('PIM_LOAD_QUERY',row['query_broadcast_ns'],'serial'),
                        ('DRAM_METADATA_READ',row['dram_read_ns'],'serial'),
                        ('PIM_BITWISE_POPCOUNT_FP_WEIGHTED_SCORE',row['pim_compute_ns'],'serial')]
            if mode == 'host':
                commands += [('PIM_RETURN_SCORE', row['score_delivery_or_control_ns'],'serial'),
                             ('HOST_GPU_ANALYTICAL_TOPK',row['selection_ns'],'serial')]
            elif mode.startswith('bank_'):
                if mode == 'bank_adaptive':
                    commands += [('PIM_THRESHOLD_SURVEY_AND_QUOTA_EXCHANGE',row['score_delivery_or_control_ns'],'serial')]
                commands += [('PIM_LOCAL_TOPK_AND_SURVEY_COMPUTE',row['selection_ns'],'serial'),
                             ('PIM_RETURN_INDICES',row['output_index_transfer_ns'],'serial')]
            else:
                commands += [('PIM_RETURN_SCORE_TO_LOGIC_DIE',row['score_delivery_or_control_ns'],'stream'),
                             ('NEAR_MEMORY_GLOBAL_TOPK_STREAM',row['selection_ns'],'stream'),
                             ('NEAR_MEMORY_RETURN_INDICES',row['output_index_transfer_ns'],'serial')]
            commands += [('HOST_AND_CONTROL_SYNC',row['sync_ns'],'serial')]
            for command, duration, overlap in commands:
                writer.writerow(dict(mode=mode,command=command,duration_ns=duration,
                                     overlap_group=overlap,
                                     source='Analytical phase trace; NOT command timing'))
