import torch


def save_processed_dataset(dataset, output_path):
    """
    Save the processed dataset to disk.
    
    Args:
        dataset (SpliceGraphDataset): The dataset to save
        output_path (str): Path to save the dataset
    """
    print(f"Saving dataset with {len(dataset)} graphs to {output_path}...")
    
    # Process and collect all graphs
    all_data = []
    for i in range(len(dataset)):
        print(f"Processing graph {i+1}/{len(dataset)}")
        try:
            sample = dataset[i]
            all_data.append(sample)
        except Exception as e:
            print(f"Error processing graph {i}: {str(e)}")
    
    # Save to disk
    torch.save(all_data, output_path)
    print(f"Dataset saved successfully to {output_path}")


def load_processed_dataset(input_path):
    """
    Load a previously saved dataset.
    
    Args:
        input_path (str): Path to the saved dataset file
    
    Returns:
        list: List of processed graph dictionaries
    """
    print(f"Loading dataset from {input_path}...")
    all_data = torch.load(input_path)
    print(f"Loaded {len(all_data)} graphs")
    return all_data