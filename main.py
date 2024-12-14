from KAN_Implementation import *
import torch
import os
import warnings


# CUDA settings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


def main():
    print("Starting KAN scaling analysis...")
    analyzer = KANScalingAnalysis()

    # Run analysis and create visualization
    results_df = analyzer.run_analysis()

    if len(results_df) > 0:
        analyzer.visualize_scaling(results_df)
        print("\nAnalysis complete!")
    else:
        print("\nNo results were collected. Check the error messages above.")

    return results_df


if __name__ == "__main__":
    main()
