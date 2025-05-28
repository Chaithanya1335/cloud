from setuptools import find_packages,setup
from typing import List


def get_requiremets(path:str)->List[str]:
    
    with open(path) as file:
        requirement = file.readlines()

        requirement = [req.replace("\n",'') for req in requirement if req != '-e .']
    
    return requirement


setup(
    name="Cloud",
    version='0.0.1',
    author="Gnana Chaithanya",
    author_email='m.gnanachaithanya12@gmail.com',
    packages=find_packages(),
    install_requires = get_requiremets('requirements.txt') 
)
