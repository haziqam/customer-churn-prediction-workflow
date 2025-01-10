import os

def create_directory_if_not_exists(directory):
    """
    Creates a directory if it doesn't exist.
    
    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# Example usage
if __name__ == "__main__":
    create_directory_if_not_exists("../artifacts")
