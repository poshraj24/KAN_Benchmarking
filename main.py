from KAN_Implementation import *
import torch
import os
import warnings
import sys
import torch.multiprocessing as tmp


def setup_cuda():
    """Setup and optimize CUDA settings"""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
    warnings.filterwarnings("ignore")

    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        # Set device and clear cache
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()

        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
        )
    else:
        print("CUDA is not available. Using CPU.")


def main():
    try:
        print("Starting KAN scaling analysis...")
        setup_cuda()

        # Initialize the analyzer
        analyzer = KANScalingAnalysis()

        # Run analysis and get results
        results_df = analyzer.run_analysis()

        if len(results_df) > 0:
            # Create visualizations
            KANScalingAnalysis.visualize_scaling(results_df)

            # Print summary statistics
            print("\nAnalysis Summary:")
            print(f"Total configurations processed: {len(results_df)}")
            print(f"Average Peak Memory: {results_df['peak_memory_gb'].mean():.2f} GB")
            print(f"Maximum Peak Memory: {results_df['peak_memory_gb'].max():.2f} GB")
            print(f"Average Final Loss: {results_df['final_loss'].mean():.4f}")

            print("\nAnalysis complete!")
            print("Results saved to: kan_scaling_metrics.csv")
            print("Memory statistics saved to: memory_stats_detailed.csv")
            print("Visualizations saved to: kan_scaling_analysis.png")
        else:
            print("\nNo results were collected. Check the error messages above.")

        return results_df

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Initialize multiprocessing
    tmp.set_start_method("spawn", force=True)

    try:
        results = main()
        if results is not None:
            print("Analysis completed successfully.")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
