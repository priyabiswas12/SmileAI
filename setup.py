#to build application as a package itself

from setuptools import find_packages,setup
from typing import List



HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements





setup( #metadata abt entire project
name="MeshSegDental",
version='0.0.1',
author="priyabiswas12",
author_email="priyabiswas12@gmail.com",
packages=find_packages(), #this function can find any folder as a package that contains the __init__.py file
#install_requires=["pandas", "numpy", "seaborn"],
install_requires=get_requirements('requirements.txt')

)