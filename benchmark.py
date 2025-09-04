#!/usr/bin/env python3
"""
MPI Performance Benchmarking Script for Sparse Matrix Multiplication
Runs ./Q1 with different process counts and measures computation time
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def parse_test_cases(filename):
    """Parse test cases from the input file"""
    test_cases = []
    
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    # Split by test case comments
    parts = content.split('# --- Test Case')
    
    # Skip the first part (just the number 28)
    for i, part in enumerate(parts[1:], 1):
        lines = part.strip().split('\n')
        
        # Find the Input: line
        input_start = -1
        output_start = -1
        
        for j, line in enumerate(lines):
            if line.strip() == 'Input:':
                input_start = j + 1
            elif line.strip() == 'Output:':
                output_start = j
                break
        
        if input_start >= 0 and output_start >= 0:
            input_lines = lines[input_start:output_start]
            # Remove empty lines
            input_lines = [line for line in input_lines if line.strip()]
            
            if input_lines:
                test_cases.append({
                    'case_num': i,
                    'input': '\n'.join(input_lines)
                })
    
    return test_cases

def get_matrix_dimensions(input_text):
    """Extract matrix dimensions from input"""
    lines = input_text.strip().split('\n')
    if lines:
        dimensions = lines[0].split()
        if len(dimensions) >= 3:
            N, M, P = map(int, dimensions)
            return N, M, P
    return None, None, None

def run_mpi_command(np_count, input_text, timeout=60):
    """Run MPI command with given process count and input"""
    cmd = ['mpirun', '-np', str(np_count), './Q1']
    
    try:
        start_time = time.time()
        
        # Run the command with input (compatible with older Python versions)
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Convert input text to bytes for older Python compatibility
        input_bytes = input_text.encode('utf-8')
        
        try:
            # For Python 3.3+
            stdout, stderr = process.communicate(input=input_bytes, timeout=timeout)
        except TypeError:
            # For older Python versions without timeout parameter
            stdout, stderr = process.communicate(input=input_bytes)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        # Decode output from bytes to string
        stdout = stdout.decode('utf-8') if stdout else ''
        stderr = stderr.decode('utf-8') if stderr else ''
        
        if process.returncode == 0:
            return {
                'success': True,
                'time': computation_time,
                'stdout': stdout,
                'stderr': stderr
            }
        else:
            return {
                'success': False,
                'time': computation_time,
                'stdout': stdout,
                'stderr': stderr,
                'returncode': process.returncode
            }
    
    except Exception as e:
        # Handle timeout and other exceptions
        if "TimeoutExpired" in str(type(e)) or "timeout" in str(e).lower():
            try:
                process.kill()
            except:
                pass
            return {
                'success': False,
                'time': timeout,
                'error': 'Timeout'
            }
        else:
            return {
                'success': False,
                'time': 0,
                'error': str(e)
            }

def main():
    # Configuration
    input_file = 'sparse_testcases.txt'
    output_file = 'mpi_performance_results2.csv'
    process_counts = [1, 2, 4, 8, 16, 32]  # Different MPI process counts to test
    # process_counts = [1, 2, 4]
    # Check if Q1 executable exists
    if not os.path.exists('./Q1'):
        print("Error: ./Q1 executable not found!")
        print("Please compile your program first.")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        sys.exit(1)
    
    # Parse test cases
    print("Parsing test cases...")
    test_cases = parse_test_cases(input_file)
    print(f"Found {len(test_cases)} test cases")
    
    # Prepare output file
    with open(output_file, 'w') as f:
        f.write("TestCase,ProcessCount,N,M,P,InputSize,ComputationTime(s),Success,Error\n")
    
    # Run benchmarks
    total_runs = len(test_cases) * len(process_counts)
    current_run = 0
    
    for test_case in test_cases:
        case_num = test_case['case_num']
        input_text = test_case['input']
        
        # Get matrix dimensions
        N, M, P = get_matrix_dimensions(input_text)
        input_size = f"{N}x{M}x{P}" if N and M and P else "Unknown"
        
        print(f"\nRunning Test Case {case_num} (Dimensions: {input_size})")
        print("-" * 50)
        
        for np_count in process_counts:
            current_run += 1
            print(f"  Progress: {current_run}/{total_runs} - Running with {np_count} processes... ", end="", flush=True)
            
            result = run_mpi_command(np_count, input_text)
            
            # Write results to CSV
            with open(output_file, 'a') as f:
                if result['success']:
                    f.write(f"{case_num},{np_count},{N},{M},{P},{input_size},{result['time']:.6f},True,\n")
                    print(f"✓ {result['time']:.3f}s")
                else:
                    error_msg = result.get('error', f"ReturnCode:{result.get('returncode', 'Unknown')}")
                    # Escape commas in error message
                    error_msg = error_msg.replace(',', ';')
                    f.write(f"{case_num},{np_count},{N},{M},{P},{input_size},{result['time']:.6f},False,{error_msg}\n")
                    print(f"✗ Failed ({error_msg})")
    
    print(f"\n{'='*60}")
    print(f"Benchmarking completed!")
    print(f"Results saved to: {output_file}")
    print(f"Total runs: {total_runs}")
    print(f"{'='*60}")
    
    # Generate summary
    print("\nGenerating summary...")
    try:
        import pandas as pd
        df = pd.read_csv(output_file)
        
        print("\nSummary Statistics:")
        print("-" * 30)
        
        # Success rate by process count
        success_rate = df.groupby('ProcessCount')['Success'].mean() * 100
        print("Success Rate by Process Count:")
        for pc in process_counts:
            if pc in success_rate.index:
                print(f"  {pc:2d} processes: {success_rate[pc]:6.1f}%")
        
        # Output correctness rate by process count
        if 'OutputMatch' in df.columns:
            correct_rate = df[df['Success'] == True].groupby('ProcessCount')['OutputMatch'].mean() * 100
            print("\nOutput Correctness Rate (among successful runs):")
            for pc in process_counts:
                if pc in correct_rate.index:
                    print(f"  {pc:2d} processes: {correct_rate[pc]:6.1f}%")
        
        # Average computation time by process count (successful runs with correct output only)
        successful_correct_runs = df[(df['Success'] == True) & (df['OutputMatch'] == True)]
        if not successful_correct_runs.empty:
            avg_time = successful_correct_runs.groupby('ProcessCount')['ComputationTime(s)'].mean()
            print("\nAverage Computation Time (successful + correct runs only):")
            for pc in process_counts:
                if pc in avg_time.index:
                    print(f"  {pc:2d} processes: {avg_time[pc]:8.4f}s")
        
        # Test case statistics
        print(f"\nTest Case Statistics:")
        print(f"  Total test cases: {df['TestCase'].nunique()}")
        print(f"  Total runs: {df.shape[0]}")
        print(f"  Successful runs: {successful_runs.shape[0]} ({successful_runs.shape[0]/df.shape[0]*100:.1f}%)")
        if 'OutputMatch' in df.columns:
            correct_runs = df[(df['Success'] == True) & (df['OutputMatch'] == True)]
            print(f"  Correct runs: {correct_runs.shape[0]} ({correct_runs.shape[0]/df.shape[0]*100:.1f}%)")
        
    except ImportError:
        print("Note: Install pandas for detailed summary statistics (pip install pandas)")
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()
