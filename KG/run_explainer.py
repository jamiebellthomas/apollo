#!/usr/bin/env python3
"""
Simple script to run the HeteroGNN5Explainer with different configurations.
This demonstrates how to use the explainer for different analysis scenarios.
"""

import os
import sys
from HeteroGNN5Explainer import main, CONFIG

def run_basic_explanation():
    """Run the explainer with default settings for positive predictions only."""
    print("Running basic explanation analysis...")
    main()
    print("Basic explanation completed!")

def run_comprehensive_explanation():
    """Run the explainer with comprehensive settings for all predictions."""
    print("Running comprehensive explanation analysis...")
    
    # Modify config for comprehensive analysis
    CONFIG["ONLY_POSITIVE_LABELS"] = False
    CONFIG["TOPK"] = 10
    CONFIG["OUT_DIR"] = "./heterognn5_explanations_comprehensive"
    CONFIG["DO_IMPACT_CHECK"] = True
    
    main()
    print("Comprehensive explanation completed!")

def run_focused_explanation():
    """Run the explainer focused on high-confidence predictions."""
    print("Running focused explanation analysis...")
    
    # Modify config for focused analysis
    CONFIG["ONLY_POSITIVE_LABELS"] = False
    CONFIG["THRESHOLD_LOGIT"] = 0.0  # Only analyze predictions with logit >= 0
    CONFIG["TOPK"] = 3
    CONFIG["OUT_DIR"] = "./heterognn5_explanations_focused"
    CONFIG["DO_IMPACT_CHECK"] = True
    
    main()
    print("Focused explanation completed!")

def run_uncertainty_analysis():
    """Run the explainer with Monte Carlo dropout for uncertainty analysis."""
    print("Running uncertainty analysis...")
    
    # Modify config for uncertainty analysis
    CONFIG["ONLY_POSITIVE_LABELS"] = True
    CONFIG["MC_DROPOUT_EVAL"] = True
    CONFIG["TOPK"] = 5
    CONFIG["OUT_DIR"] = "./heterognn5_explanations_uncertainty"
    CONFIG["DO_IMPACT_CHECK"] = False  # Disable for faster execution
    
    main()
    print("Uncertainty analysis completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analysis_type = sys.argv[1].lower()
        
        if analysis_type == "basic":
            run_basic_explanation()
        elif analysis_type == "comprehensive":
            run_comprehensive_explanation()
        elif analysis_type == "focused":
            run_focused_explanation()
        elif analysis_type == "uncertainty":
            run_uncertainty_analysis()
        else:
            print(f"Unknown analysis type: {analysis_type}")
            print("Available options: basic, comprehensive, focused, uncertainty")
    else:
        print("Running default basic explanation...")
        print("Usage: python run_explainer.py [basic|comprehensive|focused|uncertainty]")
        run_basic_explanation()
