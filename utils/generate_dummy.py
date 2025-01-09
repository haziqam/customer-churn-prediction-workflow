import pandas as pd
import numpy as np
from faker import Faker

# Load the dataset
file_path = 'workspace/dataset.csv'
data = pd.read_csv(file_path)

# Initialize Faker for generating new IDs
fake = Faker()

# Function to simulate new data
def generate_synthetic_data(original_df, num_samples=1000):
    synthetic_data = pd.DataFrame()
    
    # Generate new customer IDs
    synthetic_data['customerID'] = [fake.unique.uuid4()[:8] for _ in range(num_samples)]
    
    # Randomly sample existing data to base new rows on
    sampled_data = original_df.sample(num_samples, replace=True).reset_index(drop=True)
    
    # Perturb numerical features
    synthetic_data['tenure'] = sampled_data['tenure'] + np.random.randint(-3, 4, num_samples)
    synthetic_data['MonthlyCharges'] = sampled_data['MonthlyCharges'] * np.random.uniform(0.9, 1.1, num_samples)
    synthetic_data['SeniorCitizen'] = sampled_data['SeniorCitizen']
    
    # Copy and shuffle categorical features
    for col in ['gender', 'Partner', 'Dependents', 'InternetService', 'Contract', 
                'PaperlessBilling', 'PaymentMethod', 'Churn']:
        synthetic_data[col] = sampled_data[col]
    
    # Adjust some categorical columns with random flipping
    synthetic_data['PhoneService'] = np.where(np.random.rand(num_samples) > 0.5, 'Yes', 'No')
    synthetic_data['MultipleLines'] = np.where(np.random.rand(num_samples) > 0.5, 'Yes', 'No')
    
    # Handle any potential inconsistencies
    synthetic_data['tenure'] = synthetic_data['tenure'].clip(lower=0)
    synthetic_data['MonthlyCharges'] = synthetic_data['MonthlyCharges'].clip(lower=0)
    
    return synthetic_data

# Generate synthetic data
synthetic_df = generate_synthetic_data(data, num_samples=1000)

# Save the synthetic data
synthetic_df.to_csv('workspace/dummy_data2.csv', index=False)
