#!/usr/bin/env python3
"""
Merge Stock Files Utility
Combines all individual stock checkpoint files into a single Excel file.
Filters out placeholder records (POSSIBLY_BLOCKED, NO_ARTICLES_FOUND).

Usage:
    python merge_stock_files.py                    # Use default output folder
    python merge_stock_files.py --input ./output   # Specify input folder
    python merge_stock_files.py --keep-placeholders  # Include placeholder records
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd


def merge_stock_files(
    input_dir: str = "output",
    output_file: str = None,
    keep_placeholders: bool = False,
    sort_by: str = "Stock_Code"
):
    """
    Merge all checkpoint Excel files into one.
    
    Args:
        input_dir: Directory containing checkpoint_*.xlsx files
        output_file: Output filename (auto-generated if None)
        keep_placeholders: If True, include POSSIBLY_BLOCKED and NO_ARTICLES_FOUND records
        sort_by: Column to sort the final output by
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist!")
        return None
    
    # Find all checkpoint files
    checkpoint_files = list(input_path.glob("checkpoint_*.xlsx"))
    
    if not checkpoint_files:
        print(f"❌ No checkpoint files found in '{input_dir}'")
        return None
    
    print(f"📁 Found {len(checkpoint_files)} checkpoint files in '{input_dir}'")
    
    # Track which stocks we've seen (to avoid duplicates if retried)
    stock_data = {}  # stock_code -> (records, file_timestamp)
    
    all_dfs = []
    files_processed = 0
    records_total = 0
    
    for file_path in sorted(checkpoint_files):
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            
            if df.empty:
                print(f"  ⚠️ Skipping empty file: {file_path.name}")
                continue
            
            # Get stock code from the data
            if 'Stock_Code' in df.columns and len(df) > 0:
                stock_code = df['Stock_Code'].iloc[0]
                stock_name = df['Stock_Name'].iloc[0] if 'Stock_Name' in df.columns else 'Unknown'
                
                # Extract timestamp from filename 
                # New format: checkpoint_1211_HK_BYD_20260201_234645.xlsx
                # Old format: checkpoint_1211_HK_20260201_234645.xlsx
                filename = file_path.stem
                try:
                    # Parse timestamp from filename (always last 2 parts)
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        timestamp_str = '_'.join(parts[-2:])  # 20260201_234645
                        file_timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    else:
                        file_timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)
                except:
                    file_timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                # Check if we already have data for this stock
                if stock_code in stock_data:
                    existing_timestamp = stock_data[stock_code][1]
                    if file_timestamp > existing_timestamp:
                        # Newer file, replace
                        print(f"  🔄 Replacing {stock_code} ({stock_name}) with newer data from {file_path.name}")
                        stock_data[stock_code] = (df, file_timestamp)
                    else:
                        print(f"  ⏭️ Skipping older file for {stock_code}: {file_path.name}")
                else:
                    stock_data[stock_code] = (df, file_timestamp)
                    print(f"  ✓ Loaded {stock_code} ({stock_name}): {len(df)} records from {file_path.name}")
            else:
                # No stock code column, just add all records
                all_dfs.append(df)
                print(f"  ✓ Loaded {len(df)} records from {file_path.name}")
            
            files_processed += 1
            
        except Exception as e:
            print(f"  ❌ Error reading {file_path.name}: {e}")
            continue
    
    # Combine all DataFrames
    for stock_code, (df, _) in stock_data.items():
        all_dfs.append(df)
    
    if not all_dfs:
        print("❌ No valid data found in any files!")
        return None
    
    # Merge all DataFrames
    merged_df = pd.concat(all_dfs, ignore_index=True)
    records_total = len(merged_df)
    
    print(f"\n📊 Total records before filtering: {records_total}")
    
    # Filter out placeholders if requested
    if not keep_placeholders:
        placeholder_titles = ['POSSIBLY_BLOCKED', 'NO_ARTICLES_FOUND', 'NO ARTICLES FOUND']
        
        # Count placeholders
        blocked_count = len(merged_df[merged_df['News_Title'] == 'POSSIBLY_BLOCKED'])
        no_articles_count = len(merged_df[merged_df['News_Title'].isin(['NO_ARTICLES_FOUND', 'NO ARTICLES FOUND'])])
        
        # Filter
        merged_df = merged_df[~merged_df['News_Title'].isin(placeholder_titles)]
        
        print(f"  🚫 Removed {blocked_count} POSSIBLY_BLOCKED records")
        print(f"  🚫 Removed {no_articles_count} NO_ARTICLES_FOUND records")
        print(f"  ✅ Real articles: {len(merged_df)}")
    
    # Remove duplicates based on URL
    if 'News_URL' in merged_df.columns:
        before_dedup = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['News_URL'], keep='first')
        dups_removed = before_dedup - len(merged_df)
        if dups_removed > 0:
            print(f"  🔄 Removed {dups_removed} duplicate URLs")
    
    # Sort
    if sort_by in merged_df.columns:
        merged_df = merged_df.sort_values(by=[sort_by, 'News_Date'], ascending=[True, False])
    
    # Generate output filename
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"merged_news_{timestamp}.xlsx"
    
    output_path = Path(input_dir) / output_file
    
    # Save
    try:
        # Clean up data for Excel
        for col in merged_df.columns:
            merged_df[col] = merged_df[col].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else x
            )
            try:
                merged_df[col] = merged_df[col].apply(
                    lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo else x
                )
            except:
                pass
        
        merged_df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"\n✅ Merged file saved: {output_path}")
        print(f"   Total records: {len(merged_df)}")
        
        # Print stock summary
        if 'Stock_Code' in merged_df.columns:
            stock_counts = merged_df['Stock_Code'].value_counts()
            print(f"\n📈 Articles per stock:")
            for stock, count in stock_counts.items():
                print(f"   {stock}: {count} articles")
        
        return str(output_path)
        
    except Exception as e:
        print(f"❌ Error saving merged file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge stock checkpoint files into a single Excel file"
    )
    parser.add_argument(
        '--input', '-i',
        default='output',
        help='Input directory containing checkpoint_*.xlsx files (default: output)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output filename (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--keep-placeholders',
        action='store_true',
        help='Include POSSIBLY_BLOCKED and NO_ARTICLES_FOUND records'
    )
    parser.add_argument(
        '--sort-by',
        default='Stock_Code',
        help='Column to sort by (default: Stock_Code)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MERGE STOCK FILES UTILITY")
    print("=" * 60)
    
    result = merge_stock_files(
        input_dir=args.input,
        output_file=args.output,
        keep_placeholders=args.keep_placeholders,
        sort_by=args.sort_by
    )
    
    if result:
        print("\n✅ Merge completed successfully!")
    else:
        print("\n❌ Merge failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
