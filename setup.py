from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.read().splitlines()
    
    # Remove '-e .' if present and empty lines
    requirements = [req for req in requirements if req and not req.startswith('-e')]
    
    return requirements

setup(
    name="mlproject",
    version="1.0",
    author="Abhinav",
    author_email="xyz@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
