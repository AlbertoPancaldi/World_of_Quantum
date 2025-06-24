"""
Metrics Collection and Analysis

Collects, analyzes, and aggregates performance metrics from
transpilation comparisons. Provides statistical analysis and
summary reporting capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class MetricsCollector:
    """
    Collects and analyzes transpilation performance metrics.
    
    Provides comprehensive analysis of custom vs stock transpilation
    results including statistical summaries and trend analysis.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.results_history = []
        self.summary_stats = {}
        
    def collect_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Collect batch transpilation results.
        
        Args:
            results: List of transpilation comparison results
        """
        self.results_history.extend(results)
        self._update_summary_stats()
        
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Convert collected results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with transpilation metrics
        """
        if not self.results_history:
            return pd.DataFrame()
        
        rows = []
        for result in self.results_history:
            if not result.get('comparison') or not result['comparison'].get('comparison_valid'):
                continue
                
            row = {
                'circuit_name': result['circuit_name'],
                'num_qubits': result['input_stats']['num_qubits'],
                'input_depth': result['input_stats']['depth'],
                'input_cx_count': result['input_stats']['cx_count'],
                
                # Stock results
                'stock_success': result['stock']['success'],
                'stock_cx_count': result['stock']['stats'].get('cx_count', 0),
                'stock_depth': result['stock']['stats'].get('depth', 0),
                'stock_compile_time': result['stock']['compile_time'],
                
                # Custom results
                'custom_success': result['custom']['success'] if result['custom'] else False,
                'custom_cx_count': result['custom']['stats'].get('cx_count', 0) if result['custom'] else 0,
                'custom_depth': result['custom']['stats'].get('depth', 0) if result['custom'] else 0,
                'custom_compile_time': result['custom']['compile_time'] if result['custom'] else 0,
                
                # Comparison metrics
                'cx_reduction_percent': result['comparison']['cx_reduction_percent'],
                'depth_reduction_percent': result['comparison']['depth_reduction_percent'],
                'compile_time_ratio': result['comparison']['compile_time_ratio'],
                'absolute_cx_reduction': result['comparison']['absolute_cx_reduction'],
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all collected results.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.get_metrics_dataframe()
        
        if df.empty:
            return {'error': 'No valid results to analyze'}
        
        # Filter successful comparisons only
        successful_df = df[df['custom_success'] & df['stock_success']]
        
        if successful_df.empty:
            return {'error': 'No successful comparisons found'}
        
        summary = {
            'total_circuits': len(df),
            'successful_comparisons': len(successful_df),
            'success_rate': len(successful_df) / len(df) * 100,
            
            # CX reduction statistics
            'cx_reduction': {
                'mean': successful_df['cx_reduction_percent'].mean(),
                'median': successful_df['cx_reduction_percent'].median(),
                'std': successful_df['cx_reduction_percent'].std(),
                'min': successful_df['cx_reduction_percent'].min(),
                'max': successful_df['cx_reduction_percent'].max(),
                'positive_improvements': (successful_df['cx_reduction_percent'] > 0).sum(),
                'improvement_rate': (successful_df['cx_reduction_percent'] > 0).mean() * 100
            },
            
            # Depth reduction statistics
            'depth_reduction': {
                'mean': successful_df['depth_reduction_percent'].mean(),
                'median': successful_df['depth_reduction_percent'].median(),
                'std': successful_df['depth_reduction_percent'].std(),
                'min': successful_df['depth_reduction_percent'].min(),
                'max': successful_df['depth_reduction_percent'].max(),
            },
            
            # Compilation time analysis
            'compile_time': {
                'custom_mean': successful_df['custom_compile_time'].mean(),
                'stock_mean': successful_df['stock_compile_time'].mean(),
                'ratio_mean': successful_df['compile_time_ratio'].mean(),
                'faster_count': (successful_df['compile_time_ratio'] < 1.0).sum(),
            },
            
            # Circuit size analysis
            'circuit_sizes': {
                'min_qubits': successful_df['num_qubits'].min(),
                'max_qubits': successful_df['num_qubits'].max(),
                'mean_qubits': successful_df['num_qubits'].mean(),
            }
        }
        
        return summary
    
    def get_size_scaling_analysis(self) -> Dict[str, Any]:
        """
        Analyze how performance scales with circuit size.
        
        Returns:
            Dictionary with scaling analysis results
        """
        df = self.get_metrics_dataframe()
        successful_df = df[df['custom_success'] & df['stock_success']]
        
        if successful_df.empty:
            return {'error': 'No successful comparisons for scaling analysis'}
        
        # Group by qubit count ranges
        size_ranges = [
            (15, 25, "Small (15-25q)"),
            (26, 50, "Medium (26-50q)"),
            (51, 75, "Large (51-75q)"),
            (76, 150, "XLarge (76q+)")
        ]
        
        scaling_results = {}
        
        for min_q, max_q, label in size_ranges:
            subset = successful_df[
                (successful_df['num_qubits'] >= min_q) & 
                (successful_df['num_qubits'] <= max_q)
            ]
            
            if not subset.empty:
                scaling_results[label] = {
                    'count': len(subset),
                    'avg_cx_reduction': subset['cx_reduction_percent'].mean(),
                    'avg_compile_time_ratio': subset['compile_time_ratio'].mean(),
                    'improvement_rate': (subset['cx_reduction_percent'] > 0).mean() * 100,
                    'best_cx_reduction': subset['cx_reduction_percent'].max(),
                    'worst_cx_reduction': subset['cx_reduction_percent'].min(),
                }
        
        return scaling_results
    
    def print_summary_report(self) -> None:
        """Print a comprehensive summary report to console."""
        print("\n" + "="*60)
        print("📊 QUANTUM LAYOUT OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        # Overall statistics
        summary = self.get_summary_statistics()
        
        if 'error' in summary:
            print(f"❌ {summary['error']}")
            return
        
        print(f"\n📈 Overall Performance:")
        print(f"  Total circuits tested: {summary['total_circuits']}")
        print(f"  Successful comparisons: {summary['successful_comparisons']}")
        print(f"  Success rate: {summary['success_rate']:.1f}%")
        
        # CX reduction results
        cx_stats = summary['cx_reduction']
        print(f"\n🎯 CX Gate Reduction:")
        print(f"  Average reduction: {cx_stats['mean']:.1f}% ± {cx_stats['std']:.1f}%")
        print(f"  Median reduction: {cx_stats['median']:.1f}%")
        print(f"  Best reduction: {cx_stats['max']:.1f}%")
        print(f"  Worst reduction: {cx_stats['min']:.1f}%")
        print(f"  Circuits with improvement: {cx_stats['positive_improvements']}/{summary['successful_comparisons']} ({cx_stats['improvement_rate']:.1f}%)")
        
        # Compilation time
        time_stats = summary['compile_time']
        print(f"\n⏱️  Compilation Time:")
        print(f"  Custom average: {time_stats['custom_mean']:.3f}s")
        print(f"  Stock average: {time_stats['stock_mean']:.3f}s")
        print(f"  Speed ratio: {time_stats['ratio_mean']:.2f}x")
        print(f"  Faster compilations: {time_stats['faster_count']}")
        
        # Size scaling
        scaling = self.get_size_scaling_analysis()
        if 'error' not in scaling:
            print(f"\n📏 Performance by Circuit Size:")
            for size_label, stats in scaling.items():
                print(f"  {size_label}: {stats['avg_cx_reduction']:.1f}% avg reduction ({stats['count']} circuits)")
        
    def export_detailed_results(self, output_file: str = "results/detailed_metrics.csv") -> None:
        """
        Export detailed results to CSV file.
        
        Args:
            output_file: Output CSV file path
        """
        df = self.get_metrics_dataframe()
        
        if df.empty:
            print("⚠️  No results to export")
            return
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(output_file, index=False)
        print(f"✅ Detailed results exported to {output_file}")
        
        # Export summary statistics as well
        summary_file = output_file.replace('.csv', '_summary.json')
        import json
        with open(summary_file, 'w') as f:
            json.dump(self.get_summary_statistics(), f, indent=2)
        print(f"✅ Summary statistics exported to {summary_file}")
    
    def _update_summary_stats(self) -> None:
        """Update cached summary statistics."""
        self.summary_stats = self.get_summary_statistics()


if __name__ == "__main__":
    # Test metrics collector
    collector = MetricsCollector()
    
    # Mock some test results
    mock_results = [
        {
            'circuit_name': 'test_qv_15',
            'input_stats': {'num_qubits': 15, 'depth': 15, 'cx_count': 50},
            'stock': {
                'success': True,
                'stats': {'cx_count': 75, 'depth': 25},
                'compile_time': 0.5
            },
            'custom': {
                'success': True,
                'stats': {'cx_count': 60, 'depth': 22},
                'compile_time': 0.7
            },
            'comparison': {
                'comparison_valid': True,
                'cx_reduction_percent': 20.0,
                'depth_reduction_percent': 12.0,
                'compile_time_ratio': 1.4,
                'absolute_cx_reduction': 15
            }
        }
    ]
    
    collector.collect_results(mock_results)
    collector.print_summary_report()
