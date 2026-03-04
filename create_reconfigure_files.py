import pandas as pd
import datetime
import os
import sys

def main():
    # 1. Load Schedule
    schedule_file = 'observation_schedule.csv'
    if not os.path.exists(schedule_file):
        print(f"Error: {schedule_file} not found.")
        sys.exit(1)
        
    schedule_df = pd.read_csv(schedule_file)
    
    # Check required columns
    if 'target' not in schedule_df.columns or 'start_time' not in schedule_df.columns:
        print("Error: Schedule file missing 'target' or 'start_time' columns.")
        sys.exit(1)

    # Convert start_time to datetime (UTC)
    schedule_df['start_time'] = pd.to_datetime(schedule_df['start_time'])
    
    # Sort by time to ensure consecutive logic works
    schedule_df = schedule_df.sort_values('start_time')
    
    # 2. Parse Schedule into Blocks (Consecutive observations of the same target)
    all_blocks = []
    
    if not schedule_df.empty:
        current_target = None
        current_block = None
        
        for _, row in schedule_df.iterrows():
            tgt = row['target']
            t_start = row['start_time']
            
            if tgt != current_target:
                # New block starts
                if current_target is not None:
                    # Save previous block
                    current_block['target'] = current_target
                    all_blocks.append(current_block)
                
                # Initialize new block
                current_target = tgt
                current_block = {'start_time': t_start, 'nframes': 1}
            else:
                # Continue block
                current_block['nframes'] += 1
        
        # Append the last block
        if current_target is not None:
             current_block['target'] = current_target
             all_blocks.append(current_block)

    # 3. Load Metadata from pfs_designs files
    categories = ['CO', 'GA', 'GE']
    metadata = {} # ppc_code -> {cat, row_data, columns}
    cat_columns = {} 

    for cat in categories:
        source_file = f"pfs_designs/{cat}_summary_reconfigure.csv"
        if not os.path.exists(source_file):
             print(f"Warning: {source_file} not found. Skipping.")
             continue
             
        try:
            df = pd.read_csv(source_file)
            cols = df.columns.tolist()
            cat_columns[cat] = cols
            
            for _, row in df.iterrows():
                ppc_code = row['ppc_code']
                metadata[ppc_code] = {
                    'cat': cat,
                    'data': row.to_dict(),
                    'columns': cols
                }
        except Exception as e:
            print(f"Error reading {source_file}: {e}")
            continue

    # 4. Generate Output Rows
    output_rows = {cat: [] for cat in categories}
    processed_targets = set()

    for blk in all_blocks:
        tgt = blk['target']
        if tgt in metadata:
            processed_targets.add(tgt)
            meta = metadata[tgt]
            cat = meta['cat']
            row = meta['data'].copy()
            
            # Update info based on block
            # For CO, 1 schedule row = 2 frames, exptime = 450 * nframes
            # For others, 1 schedule row = 1 frame, exptime = 900 * nframes
            
            n_sched_rows = blk['nframes']
            if cat == 'CO':
                 nframes = n_sched_rows * 2
                 exptime = nframes * 450
            else:
                 nframes = n_sched_rows
                 exptime = nframes * 900
            
            row['ppc_nframes'] = nframes
            row['ppc_exptime'] = exptime
            
            # Update time
            utc_dt = blk['start_time']
            hst_dt = utc_dt - pd.Timedelta(hours=10)
            
            has_utc = 'ppc_obstime_utc' in meta['columns']
            has_hst = 'ppc_obstime' in meta['columns']
            
            if has_utc:
                row['ppc_obstime_utc'] = utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            if has_hst:
                row['ppc_obstime'] = hst_dt.strftime('%Y-%m-%dT%H:%M:%S')
                
            output_rows[cat].append(row)
        else:
            print(f"Warning: Scheduled target {tgt} not found in pfs_designs summary files.")

    # 4.5 Add backup targets (unscheduled)
    # Date logic: 1 day after the last allocated observation
    if not schedule_df.empty:
        last_obs_time = schedule_df['start_time'].max()
        # UTC date + 1 day
        backup_date_utc = last_obs_time.date() + datetime.timedelta(days=1)
    else:
        backup_date_utc = datetime.date.today() + datetime.timedelta(days=1)

    backup_counts = {cat: 0 for cat in categories}

    for ppc_code, meta in metadata.items():
        if ppc_code not in processed_targets:
            cat = meta['cat']
            row = meta['data'].copy()
            
            # Update time: Keep time, change date to backup_date_utc
            if 'ppc_obstime_utc' in row and pd.notna(row['ppc_obstime_utc']):
                try:
                    # Parse original
                    orig_dt = pd.to_datetime(row['ppc_obstime_utc'])
                    # Construct new dt with backup date and original time
                    new_dt = orig_dt.replace(year=backup_date_utc.year, month=backup_date_utc.month, day=backup_date_utc.day)
                    row['ppc_obstime_utc'] = new_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                    
                    # Update HST if present (based on new UTC)
                    if 'ppc_obstime' in row:
                        hst_dt = new_dt - pd.Timedelta(hours=10)
                        row['ppc_obstime'] = hst_dt.strftime('%Y-%m-%dT%H:%M:%S')
                except Exception as e:
                    print(f"Warning: Failed to update time for backup target {ppc_code}: {e}")
            
            output_rows[cat].append(row)
            backup_counts[cat] += 1

    for cat, count in backup_counts.items():
        if count > 0:
            print(f"Added {count} backup targets for {cat} (date set to {backup_date_utc}).")

    # 5. Write Files
    for cat in categories:
        # Only write if we have data or if the source existed (but maybe empty output is desired if nothing scheduled?)
        # User wants "corresponding rows...". If nothing scheduled for a category, file might be empty or not updated.
        # But usually we want to see the file.
        if cat in cat_columns:
            out_filename = f"{cat}_summary_reconfigure.csv"
            rows = output_rows[cat]
            if rows:
                print(f"Writing {len(rows)} rows to {out_filename}...")
                out_df = pd.DataFrame(rows)
                # Enforce column order
                out_df = out_df[cat_columns[cat]]
                out_df.to_csv(out_filename, index=False)
            else:
                 print(f"No scheduled observations for {cat}. {out_filename} not created/updated.")
        else:
            # Source file didn't exist, so we couldn't have loaded anything.
            pass

if __name__ == "__main__":
    main()