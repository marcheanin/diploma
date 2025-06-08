import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_model_results(metrics, feature_importance, output_path):
    """
    Save model training and evaluation results.
    
    Args:
        metrics (dict): Dictionary containing training and evaluation metrics
        feature_importance (pd.DataFrame): DataFrame with feature importance scores
        output_path (str): Path to save the results
    """
    os.makedirs(output_path, exist_ok=True)
    
    metrics_path = os.path.join(output_path, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to: {metrics_path}")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.sort_values('importance', ascending=False),
                x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    feature_importance_plot_path = os.path.join(output_path, 'feature_importance.png')
    plt.savefig(feature_importance_plot_path)
    plt.close()
    print(f"Feature importance plot saved to: {feature_importance_plot_path}")
    
    feature_importance_path = os.path.join(output_path, 'feature_importance.csv')
    feature_importance.to_csv(feature_importance_path, index=False)
    print(f"Feature importance data saved to: {feature_importance_path}") 