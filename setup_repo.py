import os
import sys
import subprocess

def setup_repo(repo_url=None, clone_repo=False, subdirs=None, change_to_subdir=None):
    """
    Sets up a GitHub repository in the Colab environment.

    Parameters:
    - repo_url (str, optional): The HTTPS URL of the GitHub repository. If None, uses the last provided repo_url.
    - clone_repo (bool, optional): Whether to clone the repository. Defaults to False.
    - subdirs (list of str, optional): List of subdirectories to add to sys.path.
    - change_to_subdir (str, optional): Subdirectory to change the current working directory to.

    Usage example:
    setup_repo(
        repo_url='https://github.com/username/github_repo.git',
        clone_repo=True,
        subdirs=['subdirectory1', 'subdirectory2'],
        change_to_subdir='subdirectory1'
    )

    Subsequent calls can omit repo_url:
    setup_repo(
        subdirs=['another_subdir'],
        change_to_subdir='another_subdir'
    )
    """
    # Store the last used repo_url as an attribute of the function
    if not hasattr(setup_repo, 'last_repo_url'):
        setup_repo.last_repo_url = None

    if repo_url is not None:
        setup_repo.last_repo_url = repo_url
    else:
        if setup_repo.last_repo_url is None:
            raise ValueError("No repo_url provided and no previous repo_url stored.")
        repo_url = setup_repo.last_repo_url

    # Extract the repository name from the URL
    repo_name = repo_url.rstrip('.git').split('/')[-1]
    base_path = os.path.join('/content', repo_name)

    if clone_repo:
        # Check if the repository is already cloned
        if not os.path.isdir(base_path):
            # Clone the repository
            subprocess.run(['git', 'clone', repo_url])
            print(f"Repository cloned to: {base_path}")
        else:
            print(f"Repository already exists at: {base_path}")
    else:
        # Ensure the repository exists
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Repository not found at {base_path}. Please clone it first.")

    # Add the base path or specified subdirectories to sys.path
    paths_added = []
    if subdirs is None:
        if base_path not in sys.path:
            sys.path.append(base_path)
            paths_added.append(base_path)
    else:
        for subdir in subdirs:
            path = os.path.join(base_path, subdir)
            if path not in sys.path:
                sys.path.append(path)
                paths_added.append(path)

    # Change the current working directory if specified
    if change_to_subdir:
        new_dir = os.path.join(base_path, change_to_subdir)
        os.chdir(new_dir)
        print(f"Changed current working directory to: {new_dir}")
    else:
        os.chdir(base_path)
        print(f"Changed current working directory to: {base_path}")

    # Output the updated sys.path entries
    if paths_added:
        print("Added the following directories to sys.path:")
        for path in paths_added:
            print(path)
    else:
        print("No new directories were added to sys.path.")